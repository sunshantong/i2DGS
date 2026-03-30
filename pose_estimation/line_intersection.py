import torch
from typing import Optional


def make_rotation_mat(direction: torch.Tensor, up: torch.Tensor):
    xaxis = torch.cross(up, direction)
    xaxis = torch.divide(xaxis, torch.linalg.norm(xaxis, dim=-1, keepdim=True))

    yaxis = torch.cross(direction, xaxis)
    yaxis = torch.divide(yaxis, torch.linalg.norm(yaxis, dim=-1, keepdim=True))

    rotation_matrix = torch.eye(3)
    # column1
    rotation_matrix[0, 0] = xaxis[0]
    rotation_matrix[1, 0] = yaxis[0]
    rotation_matrix[2, 0] = direction[0]
    # column2
    rotation_matrix[0, 1] = xaxis[1]
    rotation_matrix[1, 1] = yaxis[1]
    rotation_matrix[2, 1] = direction[1]
    # column3
    rotation_matrix[0, 2] = xaxis[2]
    rotation_matrix[1, 2] = yaxis[2]
    rotation_matrix[2, 2] = direction[2]

    return rotation_matrix


def exclude_negatives(camera_optical_center, sample_points, dirs):
    v = camera_optical_center[None] - sample_points
    d = torch.bmm(
        v.view(v.shape[0], 1, v.shape[-1]), dirs.view(dirs.shape[0], dirs.shape[-1], 1)
    )[..., 0, 0]
    return d > 0


def compute_line_intersection(points, directions, weights=None, return_residuals=False):
    # Compute the direction cross-products
    cross_products = torch.cross(directions[:-1], directions[1:], dim=1)

    # Compute the intersection matrix
    A = cross_products
    b = torch.sum(torch.multiply(points[1:], cross_products), dim=1)

    if weights is not None:
        A = torch.multiply(A, weights[1:, None])
        b = torch.multiply(b, weights[1:])

    # breakpoint()
    parallel_vectors = (cross_products < 1.0e-7).all(dim=-1)

    non_parallel_vectors = torch.logical_not(parallel_vectors)
    A = A[non_parallel_vectors]
    b = b[non_parallel_vectors]

    # Solve the linear system of equations using the pseudo-inverse
    lstsq_results = torch.linalg.lstsq(A, b)

    # Reshape the solution to obtain the intersection points
    intersections = lstsq_results.solution

    if return_residuals:
        return intersections, lstsq_results.residuals

    if torch.count_nonzero(parallel_vectors) != 0 and torch.isnan(intersections).any():
        print("Parallel vectors")
    if torch.isnan(intersections).any():
        print("Wrong intersection")
        print(A.shape[0])
        intersections = torch.ones_like(intersections, requires_grad=True)

    return intersections


def compute_line_intersection_impl2(
    points: torch.Tensor,          # (N, D)
    directions: torch.Tensor,      # (N, D), assumed unit-length
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the least-squares intersection of N lines in D-space
    (see Eq.13 in http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf).
    points[i] is a point on line i, directions[i] its unit direction.
    """
    N, D = points.shape

    # Build each line’s projection matrix: P_i = I - n_i n_i^T
    I = torch.eye(D, device=points.device, dtype=points.dtype)
    # projs: (N, D, D)
    projs = I.unsqueeze(0) - directions[:, :, None] * directions[:, None, :]

    # If given, shape-check weights
    if weights is not None:
        weights = weights.view(N, 1, 1)  # (N,1,1)

    # Weighted sum of P_i: R = sum_i w_i * P_i
    if weights is not None:
        R = (projs * weights).sum(dim=0)
    else:
        R = projs.sum(dim=0)

    # Build q = sum_i w_i * P_i @ p_i
    # First do (N,D, D) @ (N, D,1) → (N, D, 1), then sum over i
    p_i = points.unsqueeze(-1)            # (N, D, 1)
    q_i = projs @ p_i                     # (N, D, 1)
    if weights is not None:
        q_i = q_i * weights               # (N, D, 1)
    q = q_i.sum(dim=0).squeeze(-1)        # (D,)

    # If R is (near) singular, bail out
    detR = torch.linalg.det(R)
    if torch.abs(detR) < 1e-7:
        return torch.full((D,), float("nan"), device=R.device, dtype=R.dtype)

    # Solve Rp = q
    p = torch.linalg.solve(R, q)          # (D,)

    return p


def compute_line_intersection_impl3(
    points, directions, weights=None, return_residuals=False
):
    """
    :param points: (N, 3) array of points on the lines
    :param dirs: (N, 3) array of unit direction vectors
    :returns: (3,) array of intersection point
    """
    dirs_mat = directions[:, :, None] @ directions[:, None, :]
    points_mat = points[:, :, None]
    I = torch.eye(3, dtype=points.dtype, device=points.device)

    R_matrix = I - dirs_mat
    if weights is not None:
        R_matrix = torch.multiply(R_matrix, weights[:, None, None])
    b_matrix = (I - dirs_mat) @ points_mat
    if weights is not None:
        b_matrix = torch.multiply(b_matrix, weights[:, None, None])

    # breakpoint()

    # Solve the linear system of equations using the pseudo-inverse
    lstsq_results = torch.linalg.lstsq(R_matrix.sum(dim=0), b_matrix.sum(dim=0))

    # Reshape the solution to obtain the intersection points
    intersections = lstsq_results.solution[:, 0]

    if return_residuals:
        return intersections, lstsq_results.residuals

    return intersections


def IRLS(y, X, maxiter, w_init=1, d=0.0001, tolerance=0.001):
    n, p = X.shape
    delta = torch.full((n,), d, dtype=X.dtype, device=X.device).view(1, n)
    w = torch.full((n,), w_init, dtype=X.dtype, device=X.device)
    W = torch.diag(w)
    B = torch.inverse((X.T @ W) @ X) @ ((X.T @ W) @ y)
    for _ in range(maxiter):
        _B = B
        _w = torch.abs(y[:, None] - (X @ B[:, None])).T
        w = 1.0 / torch.maximum(delta, _w)
        W = torch.diag(w[0])
        B = torch.inverse((X.T @ W) @ X) @ ((X.T @ W) @ y)
        tol = torch.sum(torch.abs(B - _B))
        print("Tolerance = %s" % tol)
        if tol < tolerance:
            return B
    return B


def compute_line_intersection_impl4(
    points, directions, weights=None, return_residuals=False
):  # IRLS-based implementation
    # Compute the direction cross-products
    cross_products = torch.cross(directions[:-1], directions[1:], dim=1)

    # Compute the intersection matrix
    A = cross_products
    b = torch.sum(torch.multiply(points[1:], cross_products), dim=1)

    if weights is not None:
        A = torch.multiply(A, weights[1:, None])
        b = torch.multiply(b, weights[1:])

    # breakpoint()
    parallel_vectors = (cross_products < 1.0e-7).all(dim=-1)

    non_parallel_vectors = torch.logical_not(parallel_vectors)
    A = A[non_parallel_vectors]
    b = b[non_parallel_vectors]

    # Solve the linear system of equations using the pseudo-inverse
    intersection = IRLS(b, A, 100)

    return intersection


def make_rotation_mat(direction: torch.Tensor, up: torch.Tensor):
    xaxis = torch.linalg.cross(up, direction)
    xaxis = torch.divide(xaxis, torch.linalg.norm(xaxis, dim=-1, keepdim=True))

    yaxis = torch.linalg.cross(direction, xaxis)
    yaxis = torch.divide(yaxis, torch.linalg.norm(yaxis, dim=-1, keepdim=True))

    rotation_matrix = torch.eye(3)
    # column1
    rotation_matrix[0, 0] = xaxis[0]
    rotation_matrix[1, 0] = yaxis[0]
    rotation_matrix[2, 0] = direction[0]
    # column2
    rotation_matrix[0, 1] = xaxis[1]
    rotation_matrix[1, 1] = yaxis[1]
    rotation_matrix[2, 1] = direction[1]
    # column3
    rotation_matrix[0, 2] = xaxis[2]
    rotation_matrix[1, 2] = yaxis[2]
    rotation_matrix[2, 2] = direction[2]

    return rotation_matrix
