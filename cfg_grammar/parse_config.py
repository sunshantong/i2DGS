from antlr4 import InputStream, CommonTokenStream
import ast
import sys
import io

from cfg_grammar.NamespaceLexer import NamespaceLexer
from cfg_grammar.NamespaceParser import NamespaceParser


def parse_config(input_text):
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()

    try:
        input_stream = InputStream(input_text)

        lexer = NamespaceLexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        parser = NamespaceParser(token_stream)

        parser.removeErrorListeners()
        lexer.removeErrorListeners()

        tree = parser.namespace()
        extracted_dict = {}

        pairs = tree.pairs().pair()
        for pair in pairs:
            try:
                key = pair.ID().getText()
                value = pair.value()

                if value.INT() is not None:
                    dict_value = int(value.INT().getText())
                elif value.FLOAT() is not None:
                    dict_value = float(value.FLOAT().getText())
                elif value.BOOL() is not None:
                    dict_value = bool(value.BOOL().getText())
                elif value.STRING() is not None:
                    dict_value = str(value.STRING().getText())[1:-1]
                else:
                    raw_text = value.getText()
                    try:
                        dict_value = ast.literal_eval(raw_text)
                    except:
                        dict_value = raw_text

                extracted_dict[key] = dict_value

            except Exception:
                continue

        return extracted_dict

    finally:
        sys.stderr = old_stderr


def safe_parse_config(input_text):
    try:
        return parse_config(input_text)
    except Exception:
        return fallback_parse_config(input_text)


def fallback_parse_config(input_text):
    config_dict = {}

    if input_text.startswith("Namespace(") and input_text.endswith(")"):
        input_text = input_text[10:-1]

    import re
    pattern = r'(\w+)\s*=\s*([^,]+)(?=,|$)'
    matches = re.findall(pattern, input_text)

    for key, value in matches:
        value = value.strip()

        if ('[' in value and ']' not in value) or (']' in value and '[' not in value):
            continue

        if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
            config_dict[key] = value[1:-1]
        elif value.lower() in ('true', 'false'):
            config_dict[key] = value.lower() == 'true'
        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            config_dict[key] = int(value)
        elif re.match(r'^-?\d+\.\d+$', value):
            config_dict[key] = float(value)
        elif value.startswith('[') and value.endswith(']'):
            try:
                config_dict[key] = ast.literal_eval(value)
            except:
                config_dict[key] = value
        else:
            config_dict[key] = value

    return config_dict


def simple_parse_config(input_text):
    if not input_text or not input_text.startswith("Namespace("):
        return {}

    content = input_text[10:-1]
    config_dict = {}

    pairs = []
    current_pair = ""
    bracket_count = 0

    for char in content + ",":
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1

        if char == ',' and bracket_count == 0:
            if current_pair.strip():
                pairs.append(current_pair.strip())
            current_pair = ""
        else:
            current_pair += char

    for pair in pairs:
        if '=' not in pair:
            continue

        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()

        try:
            if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
                config_dict[key] = value[1:-1]
            elif value.lower() in ('true', 'false'):
                config_dict[key] = value.lower() == 'true'
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                config_dict[key] = int(value)
            elif '.' in value and all(part.isdigit() or (i == 0 and part.startswith('-') and part[1:].isdigit())
                                      for i, part in enumerate(value.split('.', 1))):
                config_dict[key] = float(value)
            elif value.startswith('[') and value.endswith(']'):
                try:
                    config_dict[key] = ast.literal_eval(value)
                except:
                    config_dict[key] = value
            else:
                config_dict[key] = value
        except:
            continue

    return config_dict


def parse_config_silent(input_text):
    # Primary parsing function that tries multiple methods silently
    try:
        result = simple_parse_config(input_text)
        if result:
            return result

        result = safe_parse_config(input_text)
        if result:
            return result

        return fallback_parse_config(input_text)

    except Exception:
        return {}


if __name__ == "__main__":
    test_config = "Namespace(data_device='cuda', eval=False, images='images', model_path='output/bicycle2', render_items=['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature'], resolution=-1, sh_degree=3, source_path='./path/to/dataset/bicycle', white_background=False)"

    try:
        result = parse_config_silent(test_config)
        print("Parsing result:", result)
    except Exception as e:
        print(f"Parsing failed: {e}")