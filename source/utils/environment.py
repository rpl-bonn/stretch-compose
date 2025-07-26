from __future__ import annotations

import yaml
from utils.recursive_config import Config


def _load_environment_file(config: Config) -> dict:
    with open(config.get_subpath("environment"), encoding="UTF-8") as environment_file:
        environment_data = yaml.safe_load(environment_file)
    return environment_data


def set_key(config: Config, model: str) -> None:
    api_data = _load_environment_file(config)["api"]
    return api_data[model]["key"]

