from email.mime import image
import tensorflow as tf
import os
import pathlib
import pandas as pd
import numpy as np

from zipfile import ZipFile
from common.GoogleDrive import download_from_drive


class DataLoader:
    def __init__(self):
        self.paths = [
            (
                "data/real-vs-fake.zip",
                "https://drive.google.com/file/d/19T-ftY1EuYizoCShITz2n1i_pLeYa0Gl/view?usp=sharing",
            )
        ]

        self.check_if_datasets_are_downloaded()
        self.check_if_datasets_are_unzipped()
        self.labels = None

    def check_if_datasets_are_downloaded(self):
        for path, url in self.paths:
            if not os.path.exists(path):
                download_from_drive(path, url)  # assume all zips will be in drive

    def check_if_datasets_are_unzipped(self):
        data_path = pathlib.Path("data")

        for zip_file in data_path.glob("*.zip"):
            dir_ = os.path.splitext(zip_file)[0]

            if not os.path.exists(dir_):
                # assume name of the dir is the same as name of the zip
                with ZipFile(zip_file, "r") as zipObj:
                    zipObj.extractall(data_path)

    def read_labels(self, labels_path):
        return pd.read_csv(labels_path)

    def load_image_from_path(self, path, channels=3):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=channels)
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image

    # Normalizing the images to [-1, 1]
    def normalize(self, image):
        image = (image / 127.5) - 1
        return image

    def image_augmentation(self, image):
        pass

    def image_resizing(self, image):
        pass

    def load_image_train(self, image_file):
        self.labels = self.read_labels("data/train.csv")
        input_image = self.load_image_from_path(image_file)
        # input_image = self.image_augmentation(input_image)
        # input_image = self.image_resizing(input_image)
        input_image = self.normalize(input_image)
        return input_image, self.labels.loc[self.labels["path"] == image_file].label

    def load_image_test(self, image_file):
        print(image_file[0])
        self.labels = self.read_labels("data/test.csv")
        input_image = self.load_image_from_path(image_file)
        # input_image = self.image_resizing(input_image)
        input_image = self.normalize(input_image)
        return input_image, self.labels.loc[self.labels["path"] == image_file].label
