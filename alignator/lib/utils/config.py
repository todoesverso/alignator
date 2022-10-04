import json
from collections import defaultdict


class Config:
    _DEFAULT_CONFIG = {
        "features": 5000,
        "align": {
            "feature_retention": 0.15,
        },
        "brightandcontrast": {
            "clip_hist_percent": 1,
        },
        "histogram": {
            "grid_size": 10,
            "clip_limit": 1.5,
        },
        "resize": {
            "width": None,
            "height": None,
        },
        "vignette": {
            "sigma": 1200,
        },
        "video": {
            "framerate": 1,
            "output": "output.mp4",
        },
    }

    def __init__(self, parametrs: dict) -> None:
        self.custom_params = self._build_config_from_parametes(parametrs)

    @property
    def as_dict(self) -> dict:
        return self._DEFAULT_CONFIG | self.custom_params

    @property
    def as_str(self) -> str:
        return json.dumps(self.as_dict, default=str, indent=1)

    @property
    def default_config(self) -> dict:
        return self._DEFAULT_CONFIG

    def _build_config_from_parametes(self, parameters: dict) -> dict:
        """The idea of this method is that it will parse locals() from the main
        program with all parametrs passed (currently using typer).
        It handles two cases:
            1- --argument 123
               This will end up as { "argument": 123 }
            2- --thing-something-else 456
               This will end up as { "thing": { "something-else": 456 } }
               The idea is that "thing" should be the name of the manipulator
               class.
        """
        config = defaultdict(dict)
        for k, v in parameters.items():
            key_splited = k.split("_")
            key_clean = key_splited[0]
            if len(key_splited) > 1:
                config[key_clean]["_".join(key_splited[1:])] = v
            else:
                config[key_clean] = v

        return config
