import cv2
import tensorflow as tf



# USE A CUSTOM OpenCV FUNCTION TO READ THE IMAGE, INSTEAD OF THE STANDARD
# TensorFlow  `tf,read_file()` operation.
def _read_py_function(filename, label):
    image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
    return image_decoded, label

# USE STANDARD TensorFlow OPERATIONS TO RESIZE THE IMAGE TO FIXED SHAPE
def _resize_function(image_decoded, label):
    image_decoded.set_shape([None, None, None])
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label

filenames = ['/path/image1.jpg', '/path/image2.jpg']
labels = [0, 37, 29, 1, ...]

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype]
    ))
)

dataset = dataset.map(_resize_function)

