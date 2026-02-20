import yaml

def txt_to_yaml(
    txt_path: str,
    yaml_path: str,
    key: str = "scenario_tokens",
):
    # Read tokens from txt
    with open(txt_path, "r") as f:
        tokens = [line.strip() for line in f if line.strip()]

    # Create YAML structure
    data = {key: tokens}

    # Write YAML
    with open(yaml_path, "w") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
        )

if __name__ == "__main__":
    txt_to_yaml(
        txt_path="scenario_tokens.txt",
        yaml_path="scenario_tokens.yaml",
    )
