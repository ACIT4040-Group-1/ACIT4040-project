import yaml
from yaml.loader import SafeLoader


def get_config():
    with open('common/config.yml') as f:
        data = yaml.load(f, Loader=SafeLoader)

        return data
if __name__ == "__main__":
    get_config()