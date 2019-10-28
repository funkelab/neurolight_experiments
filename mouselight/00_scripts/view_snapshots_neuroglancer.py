from neurolight.visualizations.view_snapshot_neuroglancer import visualize_hdf5

import sys
from pathlib import Path

# script for visualizing a snapshot

filename = sys.argv[1]

voxel_size = [10,3,3]

visualize_hdf5(Path(filename), voxel_size)