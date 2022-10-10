import os
import yaml
import pathlib
import pandas as pd
import tensorflow as tf
from zipfile import ZipFile
from matplotlib import pyplot as plt
from yaml.loader import SafeLoader

from config import config
from src.common.GoogleDrive import download_from_drive


class DataLoader:
    def __init__(self):

        """
        DataLoader Class
        Downloads and unzips dataset
        Config can be set in common/configuration.yml
        Returns a generator of tensors -> (tensor(batch_size, images), tensor(batch_size, labels))

        """
        self.paths = [
            (
                "data/real-vs-fake.zip",
                "https://drive.google.com/file/d/1mlYAcWcdL5UwZ6MuPTdazsN85bV924zl/view?usp=sharing",
            )
        ]

        self.check_if_datasets_are_downloaded()
        self.check_if_datasets_are_unzipped()

        self.batch_size = config["batch_size"]

        self.limit = config["limit"]
        self.seed = config["seed"]

    def check_if_datasets_are_downloaded(self):
        for path, url in self.paths:
            if not os.path.exists(path):
                print("Could not find dataset on local machine. Downloading..")
                download_from_drive(path, url)  # assume all zips will be in drive

    def check_if_datasets_are_unzipped(self):
        data_path = pathlib.Path("data")

        for zip_file in data_path.glob("*.zip"):
            dir_ = os.path.splitext(zip_file)[0]

            if not os.path.exists(dir_):
                # assume name of the dir is the same as name of the zip
                with ZipFile(zip_file, "r") as zipObj:
                    zipObj.extractall(data_path)

    def read_labels(self, labels_path, limit=None):
        return pd.read_csv(labels_path, nrows=limit)

    def get_data(self, name="train"):
        match name:
            case "train":
                labels_path = "data/train.csv"
                load = self.load_image_train
            case "test":
                labels_path = "data/test.csv"
                load = self.load_image_test_or_val
            case "valid":
                labels_path = "data/valid.csv"
                load = self.load_image_test_or_val

        df = self.read_labels(labels_path)
        df = df.sample(frac=1, random_state=self.seed)

        if self.limit is not None:
            n_samples = int((df.shape[0] * self.limit) / 100)
            df = df.head(n=n_samples - 1)

        dataset = tf.data.Dataset.from_tensor_slices(
            (df["path"].values, df["label"].values)
        )
        dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.shuffle(buffer_size=self.batch_size)
        dataset = dataset.batch(self.batch_size)

        return dataset

    def load_image_from_path(self, path, channels=3):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=channels)
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image

    @tf.function()
    def image_augmentation(self, input_image):
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.random_crop(
                input_image,
                size=[
                    config["image_height"],
                    config["image_width"],
                    config["image_channels"],
                ],
            )

        if tf.random.uniform(()) > 0.5:

            if tf.random.uniform(()) > 0.5:
                input_image = tf.image.rot90(input_image, k=1)  # rotate 90 degrees
            else:
                input_image = tf.image.rot90(input_image, k=3)  # rotate 270 degrees

        return input_image

    def image_resizing(self, image):
        pass

    def load_image_train(self, image_file, label):
        input_image = self.load_image_from_path(image_file)
        # input_image = self.image_augmentation(input_image)
        # input_image = self.image_resizing(input_image)
        # input_image = self.normalize(input_image)
        return input_image, label

    def load_image_test_or_val(self, image_file, label):
        input_image = self.load_image_from_path(image_file)
        # input_image = self.image_resizing(input_image)
        # input_image = self.normalize(input_image)
        return input_image, label


if __name__ == "__main__":

    DL = DataLoader()
    train_dataset = DL.get_data("train")
    test_dataset = DL.get_data("test")

    for image, label in train_dataset.take(10):
        plt.imshow(image[0])
        plt.show()
