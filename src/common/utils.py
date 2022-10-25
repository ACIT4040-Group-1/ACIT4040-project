import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import shutil
import gdown
from config import config
import numpy as np



def copy_config(destination_dir):
    shutil.copy(src='config/configuration.yml', dst=destination_dir)


def download_from_drive(destination, url):
    gdown.download(url, destination, quiet=False, fuzzy=True)

#
def plot_predictions(data, model , batch = 1):
    """
        Input: validation data, model that returns a 1D result matrix
        batch: 1
    """
    batch_size = config['batch_size']
    start = (batch - 1) * batch_size
    end = start + batch_size
    result = np.round(model.predict(data), 2)
    # print(f"start: {start}, end: {end}")

    batch_pred = result[start:end]
    plt.figure(figsize=(40, 40))

    for image, label in data.take(batch):
        for i in range(10):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(image[i])
            plt.title(f"T: {label.numpy()[i]}, P: {batch_pred[i]}")
            plt.axis("off")

    plt.show()



