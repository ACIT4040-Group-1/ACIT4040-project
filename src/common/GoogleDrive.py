import gdown


def download_from_drive(destination, url):
    gdown.download(url, destination, quiet=False, fuzzy=True)
