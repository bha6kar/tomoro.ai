import yaml


def get_config(path: str = "src/main/config/config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
