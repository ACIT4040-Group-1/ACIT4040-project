import tensorflow as tf
import os
import pathlib
from zipfile import ZipFile

from common.GoogleDrive import download_from_drive


class DataLoader:
    def __init__(self):
        self.paths = [
            (
                "data/archive.zip",
                "https://drive.google.com/file/d/1xj7RbvX_rGvqBk2mFXSLVSCubCujRRh1/view?usp=sharing",
            )
        ]

        self.check_if_datasets_are_downloaded()
        self.check_if_datasets_are_unzipped()

    def check_if_datasets_are_downloaded(self):
        for path, url in self.paths:
            if not os.path.exists(path):
                download_from_drive(path, url)  # assume all zips will be in drive

    def load_image_from_path(self, path, channels=3):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=channels)

        image = tf.image.convert_image_dtype(image, tf.float32)
        return image

    def check_if_datasets_are_unzipped(self):
        data_path = pathlib.Path("data")

        for zip_file in data_path.glob("*.zip"):
            dir_ = os.path.splitext(zip_file)[0]

            if not os.path.exists(
                dir_
            ):  # assume name of the dir is the same as name of the zip
                with ZipFile(zip_file, "r") as zipObj:
                    zipObj.extractall(data_path)


# TODO:
# 1. chech if datasets are downloaded DONE
# 2. check if datasets are unzipped DONE
# 3. transform dataset to common structure
