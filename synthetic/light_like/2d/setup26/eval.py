from neurolight.tensorflow.add_loss import TENSOR_NAMES
from neurolight.tensorflow.mknet import mknet
from neurolight.tensorflow.build_synthetic_pipeline import predict
import json
import logging

from pathlib import Path
import sys

checkpoint = sys.argv[1]

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# logging.basicConfig(level=logging.DEBUG, filename="log.txt")
logging.basicConfig(level=logging.WARNING, filename="training.log")

setup_config = json.load(open("../default_config.json", "r"))
setup_config.update(json.load(open("config.json", "r")))

with open("tensor_names.json", "r") as f:
    tensor_names = json.load(f)
    logging.warning(f"{tensor_names}")

checkpoint = f"train_net_checkpoint_{checkpoint}"

if __name__ == "__main__":
    predict(setup_config["NUM_ITERATIONS"], setup_config, tensor_names, TENSOR_NAMES, checkpoint)
