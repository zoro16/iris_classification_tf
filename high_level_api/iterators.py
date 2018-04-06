import tensorflow as tf

sess = tf.Session()

# # DEFINE TRAINING AND VALIDATION DATASETS WITH THE SAME STRUCTURE
# training_dataset = tf.data.Dataset.range(100).map(
#     lambda x: x + tf.random_uniform([], -10, 10, tf.int64)
# )
# validation_dataset = tf.data.Dataset.range(50)

# # A REINITIALIZABLE ITERATOR IS DEFINED BY ITS STRUCTURE. WE COULD USE THE
# # `output_types` AND `output_shapes` PROPERTIES OF EITHER `training_dataset`
# # OR `validation_dataset` HERE, BECAUSE THEY ARE COMPATIBLE 
# iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
#                                            training_dataset.output_shapes
# )
# next_element = iterator.get_next()

# training_init_op = iterator.make_initializer(training_dataset)
# validation_init_op = iterator.make_initializer(validation_dataset)

# # RUN 20 EPOCHS IN WHICH THE TRAINING DATASET IS TRAVERSED, FOLLOWED BY THE
# # 
# for _ in range(20):
#     sess.run(training_init_op)
#     for _ in range(100):
#         sess.run(next_element)

#     sess.run(validation_init_op)
#     for _ in range(50):
#         sess.run(next_element)

#===============================================================================

# training_dataset = tf.data.Dataset.range(100).map(
#     lambda x: x + tf.random_uniform([], -10, 10, tf.int64)
# ).repeat()
# validation_dataset = tf.data.Dataset.range(50)


# # A FEEDABLE ITERATOR IS DEFINED  BY A HANDLE PLACEHOLDER AND ITS STRUCTURE. WE
# # COULD USE THE `output_types` AND `output_shapes` PROPERTIES OF EITHER `training_dataset`
# handle = tf.placeholder(tf.string, shape=[])
# iterator = tf.data.Iterator.from_string_handle(handle,
#                                                training_dataset.output_types,
#                                                training_dataset.output_shapes)
# next_element = iterator.get_next()

# # You can use feedable iterators with a variety of different kinds of iterator
# # (such as one-shot and initializable iterators).
# training_iterator = training_dataset.make_one_shot_iterator()
# validation_iterator = validation_dataset.make_initializable_iterator()

# # The `Iterator.string_handle()` method returns a tensor that can be evaluated
# # and used to feed the `handle` placeholder.
# training_handle = sess.run(training_iterator.string_handle())
# validation_handle = sess.run(validation_iterator.string_handle())

# # Loop forever, alternating between training and validation.
# while True:
#   # Run 200 steps using the training dataset. Note that the training dataset is
#   # infinite, and we resume from where we left off in the previous `while` loop
#   # iteration.
#   for _ in range(200):
#     sess.run(next_element, feed_dict={handle: training_handle})

#   # Run one pass over the validation dataset.
#   sess.run(validation_iterator.initializer)
#   for _ in range(50):
#     sess.run(next_element, feed_dict={handle: validation_handle})

#===============================================================================
