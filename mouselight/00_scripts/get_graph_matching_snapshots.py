from neurolight.transforms.swc_to_graph import swc_to_pickle
from neurolight.transforms.txt_to_graph import parse_txt
from pathlib import Path

import neurolight as nl
import gunpowder as gp
import networkx as nx
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, filename="100_speed_test.log")

from tqdm import tqdm

# Get snapshots of skeletonization, consensus, matching, and raw data without fusing


# read arrays
consensus = gp.PointsKey("CONSENSUS")
skeletonization = gp.PointsKey("SKELETONIZATION")
matched = gp.PointsKey("MATCHED")
raw = gp.ArrayKey("RAW")

voxel_size = np.array([10, 3, 3], dtype=int)

pipeline = (
    (
        nl.GraphSource(
            Path("../01_test/consensus-002.obj"),
            [consensus],
            scale=voxel_size,
            transpose=(2, 1, 0),
        ),
        nl.GraphSource(
            Path("../01_test/skeletonization_carved-002-100.obj"),
            [skeletonization],
            scale=voxel_size,
            transpose=(2, 1, 0),
        ),
        gp.N5Source(
            filename="/nrs/funke/mouselight-v2/2018-07-02/consensus-neurons-with-machine-centerpoints-labelled-as-swcs-carved.n5",
            datasets={raw: "volume"},
            array_specs={
                raw: gp.ArraySpec(
                    interpolatable=True, voxel_size=voxel_size, dtype=np.uint16
                )
            },
        ),
    )
    + gp.MergeProvider()
    + nl.TopologicalMatcher(skeletonization, consensus, matched)
    + gp.RandomLocation(ensure_nonempty=consensus, ensure_centered=True)
    + gp.PrintProfilingStats(every=100)
    + gp.Snapshot(
        output_filename="consensus_skeletonization_mapping_{}.hdf".format("{id}"),
        dataset_names={
            raw: "volumes/raw",
            consensus: "points/consensus",
            skeletonization: "points/skeletonization",
            matched: "points/matched",
        },
        every=1,
    )
)

view_size = np.array([300, 300, 300], dtype=int)
request_shape = gp.Coordinate(view_size) / gp.Coordinate(voxel_size.tolist())

request = gp.BatchRequest()
request.add(consensus, request_shape)
request.add(skeletonization, request_shape)
request.add(matched, request_shape)
request.add(raw, request_shape)

with gp.build(pipeline):
    for i in range(100):
        batch = pipeline.request_batch(request)
