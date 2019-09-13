import tensorflow as tf
import json
from funlib.learn.tensorflow.models import unet, conv_pass
import numpy as np
import sys

setup = sys.argv[1]

setup_config = json.load(open("default_config.json", "r"))
setup_config.update(json.load(open("{}/config.json".format(setup), "r")))

if __name__ == "__main__":

    INPUT_SHAPE = tuple(setup_config["INPUT_SHAPE"])
    EMBEDDING_DIMS = setup_config["EMBEDDING_DIMS"]
    NUM_FMAPS = setup_config["NUM_FMAPS"]
    FMAP_INC_FACTORS = setup_config["FMAP_INC_FACTORS"]

    raw = tf.placeholder(tf.float32, shape=INPUT_SHAPE)
    raw_batched = tf.reshape(raw, (1, 1) + INPUT_SHAPE)

    with tf.variable_scope("embedding"):
        embedding_unet = unet(
            raw_batched,
            num_fmaps=NUM_FMAPS,
            fmap_inc_factors=FMAP_INC_FACTORS,
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
        num_fmaps=EMBEDDING_DIMS,
        activation=None,
        name="embedding",
    )

    embedding_norms = tf.norm(embedding_batched[0], axis=1, keepdims=True)
    embedding_scaled = embedding_batched[0] / embedding_norms

    fg_batched = conv_pass(
        fg_unet[0], kernel_sizes=[1], num_fmaps=1, activation="sigmoid", name="fg"
    )

    output_shape_batched = embedding_scaled.get_shape().as_list()
    output_shape = tuple(
        output_shape_batched[2:]
    )  # strip the batch and channel dimension

    assert all(
        np.isclose(np.array(output_shape), np.array(setup_config["OUTPUT_SHAPE"]))
    )

    embedding = tf.reshape(embedding_scaled, (EMBEDDING_DIMS,) + output_shape)
    fg = tf.reshape(fg_batched[0], output_shape)
    gt_labels = tf.placeholder(tf.int64, shape=output_shape)

    tf.train.export_meta_graph(filename="{}/train_net.meta".format(setup))
    names = {
        "raw": raw.name,
        "embedding": embedding.name,
        "fg": fg.name,
        "gt_labels": gt_labels.name,
    }
    with open("{}/tensor_names.json".format(setup), "w") as f:
        json.dump(names, f)
