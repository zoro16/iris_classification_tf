import tensorflow as tf

sess = tf.Session()

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(lambda x: x ...)  # PARSE THE RECORD TO TENSORS
dataset = dataset.repeat()              # REPEAT THE INPUT INDEFINITELY
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()


train_filenames = ["/var/data/train1.tfrecord", "/var/data/train2.tfrecord"]
sess.run(iterator.initialize, feed_dict={filenames: train_filenames})


validation_filenames = ["/var/data/validation1.tfrecord", "/var/data/validation2.tfrecord"]
sess.run(iterator.initialize, feed_dict={filenames: validation_filenames})
