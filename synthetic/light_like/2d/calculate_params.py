import argparse
import json
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    description="Calculate some parameters from a config file."
)
parser.add_argument(
    "config", type=argparse.FileType("r"), help="The config file to analyze"
)
parser.add_argument(
    "default_config",
    nargs="?",
    default="default_config.json",
    type=argparse.FileType("r"),
    help="The default config file",
)

args = parser.parse_args()
config = json.load(args.default_config)
config.update(json.load(args.config))

alpha = config["ALPHA"]
embedding_dims = config["EMBEDDING_DIMS"]
coordinate_scale = np.array(config["COORDINATE_SCALE"])
voxel_size = np.array(config["VOXEL_SIZE"])

voxel_dist = voxel_size * coordinate_scale
logging.info("Voxel distances: {}".format(voxel_dist))

reach_voxels = alpha / voxel_dist
logging.info(
    "Maximum reach in voxels, mm: {}, {}".format(
        reach_voxels, reach_voxels / coordinate_scale
    )
)
min_seperable_neighbor_embedding_dist = alpha - min(voxel_dist)
logging.info(
    "The minimum embedding space distance s.t. neighbors are split: {}".format(
        min_seperable_neighbor_embedding_dist
    )
)
max_neighbor_difference = np.linalg.norm(
    tuple(voxel_dist)
) + 2
logging.info(
    "The maximum distance between two neighboring voxels: {}".format(
        max_neighbor_difference
    )
)

