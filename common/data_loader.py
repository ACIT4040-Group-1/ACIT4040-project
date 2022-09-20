import tensorflow as tf
import os
import pathlib
from zipfile import ZipFile

from common.GoogleDrive import download_from_drive


class DataLoader:
    def __init__(self):
        self.paths = [
            ('../data/test/archive.zip', '1g9n5SY2Ix2OAz3XntFXkilWDjeBZd4VS')
        ]

        # self.check_if_datasets_are_downloaded()
        self.check_if_datasets_are_unzipped()

    def check_if_datasets_are_downloaded(self):
        for path, token in self.paths:
            if not os.path.exists(path):
                match path:
                    case '../data/test/archive.zip':
                        download_from_drive(path, token)
                    case _:
                        self.download_dataset(path, token)

    def download_dataset(self, name, url):
        path_to_zip = tf.keras.utils.get_file(
            fname=f"{name}",
            origin=url,
            extract=True)

        path_to_zip = pathlib.Path(path_to_zip)
        path = path_to_zip.parent / name

        return path

    def load_image_from_path(self, path, channels=3):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=channels)

        image = tf.image.convert_image_dtype(image, tf.float32)
        return image

    def check_if_datasets_are_unzipped(self):
        data_path = pathlib.Path('../data')

        for zip_file in data_path.glob('*.zip'):
            with ZipFile(zip_file, 'r') as zipObj:
                zipObj.extractall('../data')


if __name__ == "__main__":
    DL = DataLoader()


# TODO:
# 1. chech if datasets are downloaded
# 2. check if datasets are unzipped
# 3. transform dataset to common structure