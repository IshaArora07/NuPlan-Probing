def txt_to_yaml(txt_path: str, yaml_path: str):
    with open(txt_path, "r") as f:
        tokens = [line.strip() for line in f if line.strip()]

    with open(yaml_path, "w") as f:
        f.write("scenario_tokens:\n")
        for token in tokens:
            f.write(f'  - "{token}"\n')


if __name__ == "__main__":
    txt_to_yaml(
        txt_path="scenario_tokens.txt",
        yaml_path="scenario_tokens.yaml",
    )
