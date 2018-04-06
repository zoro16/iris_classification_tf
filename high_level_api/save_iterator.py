import tensorflow as tf

sess = tf.Session()

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random_uniform([4]), tf.random_uniform([4, 100]))
)
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()
next1, (next2, next3) = iterator.get_next()

sess.run(iterator.initializer)

# print("next1 => {} \n next2,3 => {}".format(sess.run(next1), sess.run((next2, next3))))

# CREATE A SAVABLE OBJECT FROM ITERATOR
saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
saver = tf.train.Saver()

# SAVE THE ITERATOR STATE BY ADDING IT TO SAVEABLE OBJECTS COLLECTION
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)

#with tf.Session() as sess:
    #if should_checkpoint:
    saver.save(sess, "/models/iris_classification/models/iris/checkpoint", "models/iris_classification/models/iris/checkpoint")

# with tf.Session() as sess:
#     saver.restore("/models/iris_classification/models/iris/checkpoint")


