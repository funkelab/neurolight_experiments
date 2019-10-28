import daisy
import neuroglancer
import numpy as np
import sys
import itertools
import h5py
import networkx as nx
import json
import random
import sys

setup = sys.argv[1]
f = sys.argv[2]
s = float(sys.argv[3]) if len(sys.argv) > 3 else 1

setup_config = json.load(open("default_config.json", "r"))
setup_config.update(json.load(open("{}/config.json".format(setup), "r")))
ALPHA = setup_config["ALPHA"] * s

COORDINATE_SCALE = setup_config["COORDINATE_SCALE"]

neuroglancer.set_server_bind_address("0.0.0.0")


def o(f, *args):
    try:
        return f(*args)
    except:
        return None


raw = o(daisy.open_ds, f, "volumes/raw")
labels = o(daisy.open_ds, f, "volumes/labels")

gt_fg = o(daisy.open_ds, f, "volumes/gt_fg")
embedding = o(daisy.open_ds, f, "volumes/embedding")
fg = o(daisy.open_ds, f, "volumes/fg")
maxima = o(daisy.open_ds, f, "volumes/maxima")
gradient_embedding = o(daisy.open_ds, f, "volumes/gradient_embedding")


gradient_fg = o(daisy.open_ds, f, "volumes/gradient_fg")

emst = o(daisy.open_ds, f, "emst")
edges_u = o(daisy.open_ds, f, "edges_u")
edges_v = o(daisy.open_ds, f, "edges_v")


trees = h5py.File(f)["point_trees"]


def add(s, a, name, shader=None, visible=True):

    if shader == "rgb":
        shader = """void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    kwargs = {}

    if shader is not None:
        kwargs["shader"] = shader

    data = np.expand_dims(a.to_ndarray(), axis=0)
    if len(data.shape) == 4:
        data = np.transpose(data, axes=[1, 0, 2, 3])

    s.layers.append(
        name=name,
        layer=neuroglancer.LocalVolume(
            data=data,
            offset=a.roi.get_offset()[::-1] + (0,),
            voxel_size=a.voxel_size[::-1] + (1,),
        ),
        visible=visible,
        **kwargs
    )


def build_trees_from_mst(emst, edges_u, edges_v):
    trees = nx.DiGraph()
    for edge, u, v in zip(
        emst.to_ndarray(), edges_u.to_ndarray(), edges_v.to_ndarray()
    ):
        if edge[2] > ALPHA:
            continue
        pos_u = daisy.Coordinate(
            (u[-3:] / COORDINATE_SCALE) + ((0,) + labels.roi.get_offset())
        )
        pos_v = daisy.Coordinate(
            (v[-3:] / COORDINATE_SCALE) + ((0,) + labels.roi.get_offset())
        )
        if edge[0] not in trees.nodes:
            trees.add_node(edge[0], pos=pos_u)
        else:
            assert trees.nodes[edge[0]]["pos"] == pos_u
        if edge[1] not in trees.nodes:
            trees.add_node(edge[1], pos=pos_v)
        else:
            assert trees.nodes[edge[1]]["pos"] == pos_v
        trees.add_edge(edge[0], edge[1], d=edge[2])
    return trees


def build_trees_from_swc(swc_rows):
    trees = nx.DiGraph()
    pb = []
    pbs = {
        int(node_id): node_location
        for node_id, node_location in zip(
            tuple(swc_rows[:, 0]), tuple(swc_rows[:, 1:-1])
        )
    }
    for row in swc_rows:
        u = int(row[0])
        v = int(row[-1])

        if u == -1 or v == -1:
            continue

        pos_u = daisy.Coordinate((0,) + tuple(pbs[u]))
        pos_v = daisy.Coordinate((0,) + tuple(pbs[v]))

        if u not in trees.nodes:
            trees.add_node(u, pos=pos_u)
        else:
            assert trees.nodes[u]["pos"] == pos_u
        if v not in trees.nodes:
            trees.add_node(v, pos=pos_v)
        else:
            assert trees.nodes[v]["pos"] == pos_v

        trees.add_edge(u, v, d=np.linalg.norm(pos_u - pos_v))
    return trees
    s.layers.append(name="trees", layer=neuroglancer.AnnotationLayer(annotations=pb))


def add_trees(trees, node_id, name, visible=False):
    for i, cc_nodes in enumerate(nx.weakly_connected_components(trees)):
        cc = trees.subgraph(cc_nodes)
        mst = []
        for u, v in cc.edges():
            pos_u = np.array(cc.nodes[u]["pos"]) + 0.5
            pos_v = np.array(cc.nodes[v]["pos"]) + 0.5
            mst.append(
                neuroglancer.LineAnnotation(
                    point_a=pos_u[::-1], point_b=pos_v[::-1], id=next(node_id)
                )
            )

        s.layers.append(
            name="{}_{}".format(name, i),
            layer=neuroglancer.AnnotationLayer(annotations=mst),
            annotationColor="#{:02X}{:02X}{:02X}".format(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            ),
            visible=visible,
        )


if embedding is not None:
    embedding.materialize()
    embedding.data = (embedding.data + 1) / 2

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    if raw is not None:
        add(s, raw, "raw", visible=False)
    if labels is not None:
        add(s, labels, "labels", visible=False)
    if gt_fg is not None:
        add(s, gt_fg, "gt_fg", visible=False)
    if embedding is not None:
        add(s, embedding, "embedding", shader="rgb")
    if fg is not None:
        add(s, fg, "fg", visible=False)
    if maxima is not None:
        add(s, maxima, "maxima")
    if gradient_embedding is not None:
        add(s, gradient_embedding, "grad_embedding", shader="rgb", visible=False)
    if gradient_fg is not None:
        add(s, gradient_fg, "grad_fg", shader="rgb", visible=False)

    node_id = itertools.count(start=1)

    if emst is not None:
        mst_trees = build_trees_from_mst(emst, edges_u, edges_v)
        add_trees(mst_trees, node_id, name="MST")
    swc_trees = build_trees_from_swc(trees)
    add_trees(swc_trees, node_id, name="Tree")

print(viewer)
input("Hit ENTER to quit!")
