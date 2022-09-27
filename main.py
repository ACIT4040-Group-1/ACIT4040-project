from common.data_loader import DataLoader
import tensorflow as tf

BUFFER_SIZE = 1
BATCH_SIZE = 1
PATH = "data/real-vs-fake/test"
tf.executing_eagerly()

DL = DataLoader()
train_dataset = tf.data.Dataset.list_files(str(PATH + "/*.jpg"))
print(train_dataset)
# train_dataset = train_dataset.map(
#    DL.load_image_test, num_parallel_calls=tf.data.AUTOTUNE
# )

# train_dataset = train_dataset.shuffle(BUFFER_SIZE)
# train_dataset = train_dataset.batch(BATCH_SIZE)
"""
for i in train_dataset:
    print(len(i))
    print(type(i[0]))
    print(type(i[1]))
    print(i[1].numpy())
    break
"""
