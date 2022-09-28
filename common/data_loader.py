import os
import yaml
import pathlib
import pandas as pd
import tensorflow as tf

from zipfile import ZipFile
from yaml.loader import SafeLoader
from common.GoogleDrive import download_from_drive


class DataLoader:
    def __init__(self):
        """
        DataLoader Class
        Downloads and unzips dataset
        Config can be set in common/config.yml
        Returns a generator of tensors -> (tensor(batch_size, images), tensor(batch_size, labels))

        """
        self.paths = [
            (
                "data/real-vs-fake.zip",
                "https://drive.google.com/file/d/19T-ftY1EuYizoCShITz2n1i_pLeYa0Gl/view?usp=sharing",
            )
        ]

        self.check_if_datasets_are_downloaded()
        self.check_if_datasets_are_unzipped()

        def get_config():
            with open("common/config.yml") as f:
                data = yaml.load(f, Loader=SafeLoader)

            return data

        self.config = get_config()

        self.batch_size = self.config["batch_size"]

        self.buffer_size = (
            self.batch_size
            if self.config["buffer_size"] is None
            else self.config["buffer_size"]
        )

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

    def get_data(self, name="train"):
        match name:
            case "train":
                path = "data/real-vs-fake/train/"
                labels_path = "data/train.csv"
                load = self.load_image_train
            case "test":
                path = "data/real-vs-fake/test/"
                labels_path = "data/test.csv"
                load = self.load_image_test_or_val
            case "valid":
                path = "data/real-vs-fake/valid/"
                labels_path = "data/valid.csv"
                load = self.load_image_test_or_val

        df = self.read_labels(labels_path)

        labels = []
        for image in (images := os.listdir(path)):
            labels.append(df.loc[df.id == image.split(".")[0]].label.values[0])

        images = [path + image for image in images]

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size)

        return dataset

    def load_image_from_path(self, path, channels=3):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=channels)
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image

    # Normalizing the images to [-1, 1]
    def normalize(self, image):
        image = (image / 127.5) - 1
        return image

    @tf.function()
    def image_augmentation(self, input_image):
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.random_crop(
                input_image,
                size=[
                    self.config["image_height"],
                    self.config["image_width"],
                    self.config["image_channels"],
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
        ºinput_image = self.image_augmentation(input_image)
        # input_image = self.image_resizing(input_image)
        input_image = self.normalize(input_image)
        return input_image, label

    def load_image_test_or_val(self, image_file, label):
        input_image = self.load_image_from_path(image_file)
        # input_image = self.image_resizing(input_image)
        input_image = self.normalize(input_image)
        return input_image, label