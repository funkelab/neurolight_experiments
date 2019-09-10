import tensorflow as tf
import json
from funlib.learn.tensorflow.models import unet, conv_pass
import numpy as np

setup_config = json.load(open("../default_config.json", "r"))
setup_config.update(json.load(open("config.json", "r")))

if __name__ == "__main__":

    input_shape = tuple(setup_config["INPUT_SHAPE"])

    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1, 1) + input_shape)

    with tf.variable_scope("embedding"):
        embedding_unet = unet(
            raw_batched,
            num_fmaps=setup_config["NUM_FMAPS"],
            fmap_inc_factors=setup_config["FMAP_INC_FACTORS"],
            downsample_factors=[[2, 2], [2, 2], [2, 2]],
            kernel_size_up=[[3], [3], [3]],
            constant_upsample=True,
        )
    with tf.variable_scope("fg"):
        fg_unet = unet(
            raw_batched,
            num_fmaps=6,
            fmap_inc_factors=3,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],
            kernel_size_up=[[3], [3], [3]],
            constant_upsample=True,
        )

    embedding_batched = conv_pass(
        embedding_unet[0],
        kernel_sizes=[1],
        num_fmaps=3,
        activation=None,
        name="embedding",
    )

    fg_batched = conv_pass(
        fg_unet[0], kernel_sizes=[1], num_fmaps=1, activation="sigmoid", name="fg"
    )

    output_shape_batched = embedding_batched[0].get_shape().as_list()
    output_shape = tuple(
        output_shape_batched[2:]
    )  # strip the batch and channel dimension

    assert all(
        np.isclose(np.array(output_shape), np.array(setup_config["OUTPUT_SHAPE"]))
    )

    embedding_reshaped = tf.reshape(embedding_batched[0], (3,) + output_shape)
    embedding = tf.nn.softmax(embedding_reshaped, axis=0)
    fg = tf.reshape(fg_batched[0], output_shape)
    gt_labels = tf.placeholder(tf.int64, shape=output_shape)

    tf.train.export_meta_graph(filename="train_net.meta")
    names = {
        "raw": raw.name,
        "embedding": embedding.name,
        "fg": fg.name,
        "gt_labels": gt_labels.name,
    }
    with open("tensor_names.json", "w") as f:
        json.dump(names, f)
