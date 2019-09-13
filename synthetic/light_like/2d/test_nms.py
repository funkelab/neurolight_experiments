import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from nms import max_detection

import matplotlib.pyplot as plt

test_data = np.zeros([100, 100])
test_data[39:61, :] = 1
test_data[:, 39:61] = 1

# plt.imshow(test_data[:, :, 0])
# plt.show()


with tf.Session() as sess:
    td = tf.convert_to_tensor(test_data, dtype=tf.float16)
    td = tf.reshape(td, (1, 100, 100, 1))
    maxima, reduced_maxima = max_detection(td, window_size=[1, 5, 5], threshold=0)
    m = maxima.eval()
    rm = reduced_maxima.eval()

    plt.imshow(m[:, :])
    plt.show()
    plt.imshow(rm[:, :])
    plt.show()
