import yaml
from yaml.loader import SafeLoader


def get_config():
    with open('../config.yml') as f:
        data = yaml.load(f, Loader=SafeLoader)

        return data
