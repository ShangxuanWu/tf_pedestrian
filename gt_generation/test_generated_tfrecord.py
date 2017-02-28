import tensorflow as tf
import pdb

input_file = '/mnt/sdc1/shangxuan/train_part0.tfrecords'

# Remember to generate a file name queue of you 'train.TFRecord' file path
def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    dense_keys=['image_raw', 'label'],
    # Defaults are not specified since both keys are required.
    dense_types=[tf.string, tf.int64])

  # Convert from a scalar string tensor (whose single string has
  image = tf.decode_raw(features['image_raw'], tf.uint8)

  image = tf.reshape(image, [my_cifar.n_input])
  image.set_shape([my_cifar.n_input])

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32)
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label
  

if __name__ == "__main__":
    read_and_decode(input_file)
    pass
