from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
# w = tf.Variable(tf.random_uniform([2, 2]))
# y = tf.matmul(x, w)
# output = tf.nn.softmax(y)
# init_op = w.initializer

# with tf.Session() as sess:
#   # Run the initializer on `w`.
#   sess.run(init_op)

#   # Evaluate `output`. `sess.run(output)` will return a NumPy array containing
#   # the result of the computation.
#   print("output => {}".format(sess.run(output)))

#   # Evaluate `y` and `output`. Note that `y` will only be computed once, and its
#   # result used both to return `y_val` and as an input to the `tf.nn.softmax()`
#   # op. Both `y_val` and `output_val` will be NumPy arrays.
#   y_val, output_val = sess.run([y, output])
#   print("y_val => {}".format(y_val))
#   print("output_val => {}".format(output_val))

# =====================================================================

# Define a placeholder that expects a vector of three floating-point values,
# and a computation that depends on it.
# x = tf.placeholder(tf.float32, shape=[3])
# y = tf.square(x)

# with tf.Session() as sess:
#   # Feeding a value changes the result that is returned when you evaluate `y`.
#   print(sess.run(y, {x: [1.0, 2.0, 3.0]}))  # => "[1.0, 4.0, 9.0]"
#   print(sess.run(y, {x: [0.0, 0.0, 5.0]}))  # => "[0.0, 0.0, 25.0]"

  # Raises `tf.errors.InvalidArgumentError`, because you must feed a value for
  # a `tf.placeholder()` when evaluating a tensor that depends on it.
  # sess.run(y)

  # Raises `ValueError`, because the shape of `37.0` does not match the shape
  # of placeholder `x`.
  # sess.run(y, {x: 37.0})


# =====================================================================

# y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

# with tf.Session() as sess:
#   # Define options for the `sess.run()` call.
#   options = tf.RunOptions()
#   options.output_partition_graphs = True
#   options.trace_level = tf.RunOptions.FULL_TRACE

#   # Define a container for the returned metadata.
#   metadata = tf.RunMetadata()

#   sess.run(y, options=options, run_metadata=metadata)

#   # Print the subgraphs that executed on each device.
# #  print(metadata.partition_graphs)

#   # Print the timings of each operation that executed.
#   print(metadata.step_stats)

# =====================================================================

g_1 = tf.Graph()
with g_1.as_default():
  # Operations created in this scope will be added to `g_1`.
  c = tf.constant("Node in g_1")

  # Sessions created in this scope will run operations from `g_1`.
  sess_1 = tf.Session()

g_2 = tf.Graph()
with g_2.as_default():
  # Operations created in this scope will be added to `g_2`.
  d = tf.constant("Node in g_2")

# Alternatively, you can pass a graph when constructing a `tf.Session`:
# `sess_2` will run operations from `g_2`.
sess_2 = tf.Session(graph=g_2)

assert c.graph is g_1
assert sess_1.graph is g_1

assert d.graph is g_2
assert sess_2.graph is g_2

# Print all of the operations in the default graph.
g = tf.get_default_graph()
print(g.get_operations())
