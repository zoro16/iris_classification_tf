import tensorflow as tf

sess = tf.Session()

# dataset = tf.data.Dataset.range(5)
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()

# # Typically `result` will be the output of a model, or an optimizer's
# # training operation.
# result = tf.add(next_element, next_element)

# sess.run(iterator.initializer)
# print(sess.run(next_element))  # ==> "0"
# print(sess.run(next_element))  # ==> "2"
# print(sess.run(next_element))  # ==> "4"
# print(sess.run(next_element))  # ==> "6"
# print(sess.run(next_element))  # ==> "8"
# try:
#   sess.run(next_element)
# except tf.errors.OutOfRangeError:
#   print("End of dataset")  # ==> "End of dataset"

#===============================================================================

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()
next1, (next2, next3) = iterator.get_next()

sess.run(iterator.initializer)

print("next1 => {} \n next2,3 => {}".format(sess.run(next1), sess.run((next2, next3))))

# print(next_element)
# print(next_element)
