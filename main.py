from common.data_loader import DataLoader
import tensorflow as tf

BUFFER_SIZE = 10
BATCH_SIZE = 10
PATH = "data/real-vs-fake/train"

DL = DataLoader()
train_dataset = tf.data.Dataset.list_files(str(PATH + "/*.jpg"))
train_dataset = train_dataset.map(
    DL.load_image_train, num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
