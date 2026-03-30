from math import prod
import torch
import torch.nn as nn


class CameraDirectionPredictor(nn.Module):
    def __init__(
        self,
        image_feature_channel=256,
        image_size=(16, 16),
        pospe=8,
        featureC=256,
        fea_output=3,
    ):
        super().__init__()
        self.direction_input = 2 * pospe * 3 + 3

        self.dim_reducer1, conv_image_size1 = self.build_dimensionality_reducer(
            image_size,
            image_feature_channel,
            kernel_size=5,
        )
        self.dim_reducer2, conv_image_size2 = self.build_dimensionality_reducer(
            conv_image_size1[-1],
            image_feature_channel,
            kernel_size=4,
            num_conv2d=1,
        )

        self.in_mlpC = prod(conv_image_size2[-1]) * image_feature_channel
        layer1 = nn.Linear(self.in_mlpC, featureC)
        layer2 = nn.Linear(featureC, fea_output)

        self.mlp = nn.Sequential(
            layer1,
            nn.LayerNorm(featureC),
            nn.Dropout(p=0.2),
            layer2,
        )
        self.pospe = pospe

    def build_dimensionality_reducer(
            self,
            image_size,
            image_feature_channel,
            num_conv2d: int = 3,
            kernel_size: int = 3,
    ):
        conv_image_size = []
        convs = []
        previous_conv_size = image_size
        for i in range(num_conv2d):
            output_size = [
                self.convolution_dim_output(image_dim, kernel_size)
                for image_dim in previous_conv_size
            ]
            conv_image_size.append(output_size)
            previous_conv_size = output_size

            convs.append(nn.Conv2d(image_feature_channel, image_feature_channel, kernel_size=kernel_size))
            if i < num_conv2d - 1:
                convs.append(nn.BatchNorm2d(image_feature_channel))
            else:
                convs.append(nn.Identity())
            convs.append(nn.ReLU(inplace=True))
            convs.append(nn.Dropout2d(p=0.1))

        return nn.Sequential(*convs), conv_image_size

    @staticmethod
    def convolution_dim_output(
        dimension_size: int, kernel_size: int, stride: int = 1, padding: int = 0
    ):
        return int(((dimension_size - kernel_size + 2 * padding) / stride) + 1)

    def forward(self, image_features):
        if self.training:
            image_features = image_features + torch.randn_like(image_features) * 0.01
        conv1_output = self.dim_reducer1(image_features[None])
        conv2_output = self.dim_reducer2(conv1_output)
        first_block_result = self.mlp(conv2_output.view(conv2_output.shape[0], -1))
        return first_block_result[0]
