import tensorflow as tf

sess = tf.Session()

filenames = ["/path/file1.txt", "/path/file2.txt"]
dataset = tf.data.Dataset.from_tensor_slices(filenames)


# USE 'Dataset.flat_map()' TO TRANSFORM EACH FILE AS A SEPRATE NESTED DATASET,
# AND THEN CONCATENATE THEIR CONTENTS SEQUENTIALLY INTO A SINGLE 'flat' DATASET
# * SKIP THE HEADER
# * FILTER OUT LINES THAT STARTS WITH "#" (COMMENTS)

dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        .skip(1)
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))
    )
)
