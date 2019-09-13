from funlib.learn.tensorflow.losses.um_loss import ultrametric_loss_op
import neurolight as nl
import gunpowder as gp
import json
import numpy as np
import tensorflow as tf
import logging

from typing import List
import math

from nms import max_detection

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# logging.basicConfig(level=logging.DEBUG, filename="log.txt")
logging.basicConfig(level=logging.INFO, filename="training.log")

setup_config = json.load(open("../default_config.json", "r"))
setup_config.update(json.load(open("config.json", "r")))

# Network hyperparams
INPUT_SHAPE = setup_config["INPUT_SHAPE"]
OUTPUT_SHAPE = setup_config["OUTPUT_SHAPE"]

# Loss hyperparams
MAX_FILTER_SIZE = setup_config["MAX_FILTER_SIZE"]
MAXIMA_THRESHOLD = setup_config["MAXIMA_THRESHOLD"]
COORDINATE_SCALE = setup_config["COORDINATE_SCALE"]
ALPHA = setup_config["ALPHA"]

# Skeleton generation hyperparams
SKEL_GEN_RADIUS = setup_config["SKEL_GEN_RADIUS"]
THETAS = np.array(setup_config["THETAS"]) * math.pi
SPLIT_PS = setup_config["SPLIT_PS"]
NOISE_VAR = setup_config["NOISE_VAR"]
N_OBJS = setup_config["N_OBJS"]

# Skeleton variation hyperparams
LABEL_RADII = setup_config["LABEL_RADII"]
RAW_RADII = setup_config["RAW_RADII"]
RAW_INTENSITIES = setup_config["RAW_INTENSITIES"]

# Training hyperparams
CACHE_SIZE = setup_config["CACHE_SIZE"]
NUM_WORKERS = setup_config["NUM_WORKERS"]
SNAPSHOT_EVERY = setup_config["SNAPSHOT_EVERY"]
CHECKPOINT_EVERY = setup_config["CHECKPOINT_EVERY"]


class PrintDataTypes(gp.BatchFilter):
    def process(self, batch, request):
        for key, array in batch.arrays.items():
            print("array %s has dtype %s" % (key, array.data.dtype))


class LabelToFloat32(gp.BatchFilter):
    def __init__(self, array: gp.ArrayKey, intensities: List[float] = [1.0]):
        self.array = array
        self.intensities = intensities

    def setup(self):
        self.enable_autoskip()
        spec = self.spec[self.array].copy()
        spec.dtype = np.float32
        self.updates(self.array, spec)

    def process(self, batch, request):
        array = batch[self.array]
        intensity_data = np.zeros_like(array.data, dtype=np.float32)
        for i, label in enumerate(np.unique(array.data)):
            if label == 0:
                continue
            mask = array.data == label
            intensity_data[mask] = np.maximum(
                intensity_data[mask], self.intensities[i % len(self.intensities)]
            )
        spec = array.spec
        spec.dtype = np.float32
        batch[self.array] = gp.Array(intensity_data, spec)


with open("tensor_names.json", "r") as f:
    tensor_names = json.load(f)


emst_name = "PyFuncStateless:0"
edges_u_name = "GatherV2:0"
edges_v_name = "GatherV2_1:0"
ratio_pos_name = "PyFuncStateless_1:1"
ratio_neg_name = "PyFuncStateless_1:2"
dist_name = "Sqrt:0"
num_pos_pairs_name = "PyFuncStateless_1:3"
num_neg_pairs_name = "PyFuncStateless_1:4"


def add_loss(graph):

    # k, h, w
    embedding = graph.get_tensor_by_name(tensor_names["embedding"])

    # h, w
    fg = graph.get_tensor_by_name(tensor_names["fg"])

    # h, w
    gt_labels = graph.get_tensor_by_name(tensor_names["gt_labels"])

    # h, w
    gt_fg = tf.not_equal(gt_labels, 0, name="gt_fg")

    # h, w

    _, maxima = max_detection(
        tf.reshape(fg, (1, *OUTPUT_SHAPE, 1)),
        window_size=(1, *MAX_FILTER_SIZE),
        threshold=MAXIMA_THRESHOLD,
    )

    print(maxima.name)

    # 1, k, h, w
    embedding = tf.reshape(embedding, (1,) + tuple(embedding.get_shape().as_list()))
    # k, 1, h, w
    embedding = tf.transpose(embedding, perm=[1, 0, 2, 3])

    um_loss, emst, edges_u, edges_v, _ = ultrametric_loss_op(
        embedding,
        gt_labels,
        mask=maxima,
        coordinate_scale=COORDINATE_SCALE,
        alpha=ALPHA,
        pretrain=True,
    )

    ratio_pos = graph.get_tensor_by_name("PyFuncStateless_1:1")
    ratio_neg = graph.get_tensor_by_name("PyFuncStateless_1:2")
    dist = graph.get_tensor_by_name("Sqrt:0")
    dist = tf.Print(dist, [dist.name], "dist: ")

    # Calculate a score:

    # false positives are just ratio_neg
    # false negatives are 1 - ratio_pos

    false_pos = tf.math.cumsum(ratio_neg)
    false_neg = 1 - tf.math.cumsum(ratio_pos)
    scores = false_pos + false_neg
    best_score = tf.math.reduce_min(scores)
    best_score_index = tf.math.argmin(scores)
    best_alpha = tf.gather(dist, best_score_index)
    best_score = tf.Print(best_score, [best_score], "best_score: ")
    best_alpha = tf.Print(best_alpha, [best_alpha], "best_alpha: ")

    assert emst.name == emst_name, "{} != {}".format(emst.name, emst_name)
    assert edges_u.name == edges_u_name, "{} != {}".format(edges_u.name, edges_u_name)
    assert edges_v.name == edges_v_name, "{} != {}".format(edges_v.name, edges_v_name)

    fg_loss = tf.losses.mean_squared_error(gt_fg, fg)

    loss = um_loss + fg_loss

    tf.summary.scalar("um_loss", um_loss)
    tf.summary.scalar("fg_loss", fg_loss)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("best_alpha", best_alpha)
    tf.summary.scalar("best_score", best_score)

    summaries = tf.summary.merge_all()

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-5, beta1=0.95, beta2=0.999, epsilon=1e-8
    )

    optimizer = opt.minimize(loss)

    return (loss, optimizer)


def train(n_iterations):

    point_trees = gp.PointsKey("POINT_TREES")
    labels = gp.ArrayKey("LABELS")
    raw = gp.ArrayKey("RAW")
    gt_fg = gp.ArrayKey("GT_FG")
    embedding = gp.ArrayKey("EMBEDDING")
    fg = gp.ArrayKey("FG")
    maxima = gp.ArrayKey("MAXIMA")
    gradient_embedding = gp.ArrayKey("GRADIENT_EMBEDDING")
    gradient_fg = gp.ArrayKey("GRADIENT_FG")

    # tensorflow tensors
    emst = gp.ArrayKey("EMST")
    edges_u = gp.ArrayKey("EDGES_U")
    edges_v = gp.ArrayKey("EDGES_V")
    ratio_pos = gp.ArrayKey("RATIO_POS")
    ratio_neg = gp.ArrayKey("RATIO_NEG")
    dist = gp.ArrayKey("DIST")
    num_pos_pairs = gp.ArrayKey("NUM_POS")
    num_neg_pairs = gp.ArrayKey("NUM_NEG")

    request = gp.BatchRequest()
    request.add(raw, INPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    request.add(labels, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    request.add(point_trees, INPUT_SHAPE)

    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw, INPUT_SHAPE)
    snapshot_request.add(embedding, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request.add(fg, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request.add(gt_fg, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request.add(maxima, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request.add(
        gradient_embedding, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1))
    )
    snapshot_request.add(gradient_fg, OUTPUT_SHAPE, voxel_size=gp.Coordinate((1, 1)))
    snapshot_request[emst] = gp.ArraySpec()
    snapshot_request[edges_u] = gp.ArraySpec()
    snapshot_request[edges_v] = gp.ArraySpec()
    snapshot_request[ratio_pos] = gp.ArraySpec()
    snapshot_request[ratio_neg] = gp.ArraySpec()
    snapshot_request[dist] = gp.ArraySpec()
    snapshot_request[num_pos_pairs] = gp.ArraySpec()
    snapshot_request[num_neg_pairs] = gp.ArraySpec()

    pipeline = (
        nl.SyntheticLightLike(
            point_trees,
            dims=2,
            r=SKEL_GEN_RADIUS,
            n_obj=N_OBJS,
            thetas=THETAS,
            split_ps=SPLIT_PS,
        )
        # + gp.SimpleAugment()
        # + gp.ElasticAugment([10, 10], [0.1, 0.1], [0, 2.0 * math.pi], spatial_dims=2)
        + nl.RasterizeSkeleton(
            point_trees,
            labels,
            gp.ArraySpec(
                roi=gp.Roi((None,) * 2, (None,) * 2),
                voxel_size=gp.Coordinate((1, 1)),
                dtype=np.uint64,
            ),
        )
        + gp.Copy(labels, raw)
        + nl.GrowLabels(labels, radii=LABEL_RADII)
        + nl.GrowLabels(raw, radii=RAW_RADII)
        + LabelToFloat32(raw, intensities=RAW_INTENSITIES)
        + gp.NoiseAugment(raw, var=NOISE_VAR)
        + gp.PreCache(cache_size=CACHE_SIZE, num_workers=NUM_WORKERS)
        + gp.tensorflow.Train(
            "train_net",
            optimizer=add_loss,
            loss=None,
            inputs={tensor_names["raw"]: raw, tensor_names["gt_labels"]: labels},
            outputs={
                tensor_names["embedding"]: embedding,
                tensor_names["fg"]: fg,
                "strided_slice_1:0": maxima,
                "gt_fg:0": gt_fg,
                emst_name: emst,
                edges_u_name: edges_u,
                edges_v_name: edges_v,
                ratio_pos_name: ratio_pos,
                ratio_neg_name: ratio_neg,
                dist_name: dist,
                num_pos_pairs_name: num_pos_pairs,
                num_neg_pairs_name: num_neg_pairs,
            },
            gradients={
                tensor_names["embedding"]: gradient_embedding,
                tensor_names["fg"]: gradient_fg,
            },
            save_every=CHECKPOINT_EVERY,
            summary="Merge/MergeSummary:0",
            log_dir="tensorflow_logs",
        )
        + gp.Snapshot(
            output_filename="{iteration}.hdf",
            dataset_names={
                raw: "volumes/raw",
                labels: "volumes/labels",
                point_trees: "point_trees",
                embedding: "volumes/embedding",
                fg: "volumes/fg",
                maxima: "volumes/maxima",
                gt_fg: "volumes/gt_fg",
                gradient_embedding: "volumes/gradient_embedding",
                gradient_fg: "volumes/gradient_fg",
                emst: "emst",
                edges_u: "edges_u",
                edges_v: "edges_v",
                ratio_pos: "ratio_pos",
                ratio_neg: "ratio_neg",
                dist: "dist",
                num_pos_pairs: "num_pos_pairs",
                num_neg_pairs: "num_neg_pairs",
            },
            dataset_dtypes={maxima: np.float32, gt_fg: np.float32},
            every=SNAPSHOT_EVERY,
            additional_request=snapshot_request,
        )
        # + gp.PrintProfilingStats(every=100)
    )

    with gp.build(pipeline):
        for i in range(n_iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":
    train(setup_config["NUM_ITERATIONS"])
