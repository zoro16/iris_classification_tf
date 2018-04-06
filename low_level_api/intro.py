from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

sess = tf.Session()
# ONLY BUILD THE COMPUTATION GRAPH.
# THE FOLLOWING tf.tensor OBJECTS JUST REPRESENT THE RESULTS OF THE
# OPERATIONS THAT WELL BE RUN.
# a = tf.constant(3.0, dtype=tf.float32)
# b = tf.constant(4.0)  # also tf.float32 implicitly
# total = a + b

# print(sess.run(a))
# print(sess.run(b))
# print(sess.run(total))


# # ==================================================================

# writer = tf.summary.FileWriter('./shit')
# writer.add_graph(tf.get_default_graph())

# # ==================================================================

# vec = tf.random_uniform(shape=(3,))
# out1 = vec + 1
# out2 = vec + 2
# print(sess.run(vec))
# print(sess.run(vec))
# print(sess.run((out1, out2)))

# # ==================================================================

# # CLASS
# x = tf.placeholder(tf.float32, shape=[None, 3])
# linear_model = tf.layers.Dense(units=1)
# y = linear_model(x)


# init = tf.global_variables_initializer()
# sess.run(init)

# print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))

# # FUNCTION
# x = tf.placeholder(tf.float32, shape=[None, 3])
# y = tf.layers.dense(x, units=1)

# init = tf.global_variables_initializer()
# sess.run(init)

# print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))


# ==================================================================

# features = {
#     'sales' : [[5], [10], [8], [9]],
#     'department': ['sports', 'sports', 'gardening', 'gardening']
# }

# department_column = tf.feature_column.categorical_column_with_vocabulary_list(
#     'department', ['sports', 'gardening']
# )
# department_column = tf.feature_column.indicator_column(department_column)

# columns = [
#     tf.feature_column.numeric_column('sales'),
#     department_column
# ]

# inputs = tf.feature_column.input_layer(features, columns)


# var_init = tf.global_variables_initializer()
# table_init = tf.tables_initializer()
# sess = tf.Session()
# sess.run((var_init, table_init))

# print(sess.run(inputs))

# ==================================================================

# REGRESSION MODEL

x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32, name="x")
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32, name="y_true")

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for _ in range(1000):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

print(sess.run(y_pred))
