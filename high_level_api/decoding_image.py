# READS AN IMAGE FROM A FILE, DECODES IT INTO A DENSE TENSOR, AND RESIZES IT
# TO A FIXED SHAPE.
def _parse_func(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label

# A VECTOR OF FILENAMES
filenames = tf.constant(["/path/image1.jpg", "/path/image2.jpg"])

# `labels[i]` IS TH LABEL FOR THE IMAGE IN `filenames[i]`
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_func)
