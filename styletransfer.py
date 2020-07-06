import tensorflow as tf
import numpy as np
import PIL.Image
import time
import functools
import tensorflow_hub as hub
import numpy

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

content_path = tf.keras.utils.get_file('KidPainting.jpg', "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi.ytimg.com%2Fvi%2Fg5R6JmGd2kI%2Fmaxresdefault.jpg&f=1&nofb=1")
style_path = tf.keras.utils.get_file('CameroonianArt.jpg',"https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2F4.bp.blogspot.com%2F-mMpqbE11kxs%2FT2auiQtY62I%2FAAAAAAAAAGA%2FkbiEK6NIO_A%2Fs1600%2F32_the_flutist.jpg&f=1&nofb=1")

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape) 
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
      image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
      plt.title(title)

content_image = load_img(content_path)
style_image = load_img(style_path)

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]

file_name = 'styleimage2.png'
tensor_to_image(stylized_image).save(file_name)
