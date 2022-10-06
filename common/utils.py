import yaml
from yaml.loader import SafeLoader
import shutil


def get_config():
    with open('common/config.yml') as f:
        data = yaml.load(f, Loader=SafeLoader)

        return data
def copy_config(destination_dir):
    shutil.copy(src='common/config.yml', dst=destination_dir)


if __name__ == "__main__":
    get_config()