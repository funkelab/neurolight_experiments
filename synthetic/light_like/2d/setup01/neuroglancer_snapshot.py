import daisy
import neuroglancer
import numpy as np
import sys
import itertools
import h5py
import networkx as nx

ALPHA = 0.5

neuroglancer.set_server_bind_address("0.0.0.0")

f = sys.argv[1]
raw = daisy.open_ds(f, "volumes/raw")
labels = daisy.open_ds(f, "volumes/labels")

gt_fg = daisy.open_ds(f, "volumes/gt_fg")
embedding = daisy.open_ds(f, "volumes/embedding")
fg = daisy.open_ds(f, "volumes/fg")
maxima = daisy.open_ds(f, "volumes/maxima")
gradient_embedding = daisy.open_ds(f, "volumes/gradient_embedding")


gradient_fg = daisy.open_ds(f, "volumes/gradient_fg")

emst = daisy.open_ds(f, "emst")
edges_u = daisy.open_ds(f, "edges_u")
edges_v = daisy.open_ds(f, "edges_v")


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


def build_trees(emst, edges_u, edges_v):
    trees = nx.DiGraph()
    for edge, u, v in zip(
        emst.to_ndarray(), edges_u.to_ndarray(), edges_v.to_ndarray()
    ):
        if edge[2] > ALPHA:
            continue
        pos_u = daisy.Coordinate(u[-3:] * 100) + ((0,) + labels.roi.get_offset())
        pos_v = daisy.Coordinate(v[-3:] * 100) + ((0,) + labels.roi.get_offset())
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


def add_trees(trees, node_id):
    for i, cc_nodes in enumerate(nx.weakly_connected_components(trees)):
        cc = trees.subgraph(cc_nodes)
        mst = []
        for u, v in cc.edges():
            pos_u = cc.nodes[u]["pos"]
            pos_v = cc.nodes[v]["pos"]
            mst.append(
                neuroglancer.LineAnnotation(
                    point_a=pos_u[::-1], point_b=pos_v[::-1], id=next(node_id)
                )
            )

        s.layers.append(
            name="mst_{}".format(i), layer=neuroglancer.AnnotationLayer(annotations=mst)
        )


embedding.materialize()
mi = np.amin(embedding.data)
ma = np.amax(embedding.data)
embedding.data = (embedding.data - mi) / (ma - mi)

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add(s, raw, "raw", visible=False)
    add(s, labels, "labels", visible=False)
    add(s, gt_fg, "gt_fg", visible=False)
    add(s, embedding, "embedding", shader="rgb")
    add(s, fg, "fg", visible=False)
    add(s, maxima, "maxima")
    add(s, gradient_embedding, "grad_embedding", shader="rgb", visible=False)
    add(s, gradient_fg, "grad_fg", shader="rgb", visible=False)

    node_id = itertools.count(start=1)

    mst_trees = build_trees(emst, edges_u, edges_v)
    add_trees(mst_trees, node_id)

    pb = []
    pbs = {
        int(node_id): node_location
        for node_id, node_location in zip(tuple(trees[:, 0]), tuple(trees[:, 1:-1]))
    }
    for row in trees:
        u = int(row[0])
        v = int(row[-1])

        if u == -1 or v == -1:
            continue

        pos_u = np.array((0,) + tuple(pbs[u])) + 0.5
        pos_v = np.array((0,) + tuple(pbs[v])) + 0.5
        pb.append(
            neuroglancer.LineAnnotation(
                point_a=pos_u[::-1], point_b=pos_v[::-1], id=next(node_id)
            )
        )
    s.layers.append(name="trees", layer=neuroglancer.AnnotationLayer(annotations=pb))

print(viewer)
input("Hit ENTER to quit!")
