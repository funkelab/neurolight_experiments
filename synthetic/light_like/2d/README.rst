=========
Variables
=========

* "INPUT_SHAPE": Network input shape
* "OUTPUT_SHAPE": Network output shape
* "MAX_FILTER_SIZE": Size of the max filter kernal. Smaller means more points
* "MAXIMA_THRESHOLD": Minimum foreground confidence for a local maxima to be considered for the mst
* "COORDINATE_SCALE": How much to weight coordinates in the mst
* "ALPHA": Variable Used in um_loss. How far to push two labels apart by
* "SKEL_GEN_RADIUS": Distance between points in the generated skeletons
* "THETAS": Turning angles. Given thetas [0.5, 1], a single child node must be within 0.5 radians of the old direction. However if the skeleton branches and there are two children, then both must be within 1 radian of the old direction.
* "SPLIT_PS": The probability of having `n` children is defined by `SPLIT_PS[n-1]`
* "NOISE_VAR": The amount of noise added to the raw data
* "N_OBJS": The number of skeletons to generate per volume
* "LABEL_RADII": The radius of each label
* "RAW_RADII": The radius of each signal
* "RAW_INTENSITIES": The intensity of each signal in the raw data
* "CACHE_SIZE": Number of volumes to prefetch
* "NUM_WORKERS": Number of workers
* "SNAPSHOT_EVERY": How often to save a snapshot
* "CHECKPOINT_EVERY": How often to save a checkpoint