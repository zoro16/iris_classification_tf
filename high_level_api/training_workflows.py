import tensorflow as tf

# THE SIMPLEST WAY TO ITERATE OVER A DATASET IN MULTIPLE EPOCHS
filenames = ["/path/file1.tfrecord", "/path/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.repeat(10)
dataset = dataset.batch(32)



