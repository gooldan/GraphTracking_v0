import yaml

def load_config(config_file):
    with open(config_file) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


class ConfigReader:
    def __init__(self, config_path):
        self.__config_path = config_path
        self.cfg = load_config(config_path)