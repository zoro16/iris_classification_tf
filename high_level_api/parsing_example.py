import tensorflow as tf


# TRANSFORMS A SCALAR STRING `example_proto` INTO A PAIR OF A SCALAR STRING AND
# A SCALAR INTEGER, REPRESNTING AN IMAGE AND ITS LABEL, RESPECTIVELY
def _parse_func(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int32, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["images"], parsed_features["label"]

# CREATES A DATASET THAT READS ALL OF THE EXAMPLES FROM TWO FILES, AND EXTRACTS
# THE IMAGE AND LABEL FEATURES.
filenames = ["/path/file1.tfrecord", "/path/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
