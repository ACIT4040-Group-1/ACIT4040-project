import yaml
from yaml.loader import SafeLoader
import shutil


def get_config():
    with open('config/configuration.yml') as f:
        data = yaml.load(f, Loader=SafeLoader)

        return data

def copy_config(destination_dir):
    shutil.copy(src='config/configuration.yml', dst=destination_dir)


if __name__ == "__main__":
    get_config()


def download_from_drive(destination, url):
    gdown.download(url, destination, quiet=False, fuzzy=True)
