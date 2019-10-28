from neurolight.tensorflow.add_loss import TENSOR_NAMES
from neurolight.tensorflow.mknet import mknet
from neurolight.tensorflow.build_synthetic_pipeline import train
import json
import logging

from pathlib import Path

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# logging.basicConfig(level=logging.DEBUG, filename="log.txt")
logging.basicConfig(level=logging.WARNING, filename="training.log")

setup_config = json.load(open("../default_config.json", "r"))
setup_config.update(json.load(open("config.json", "r")))


if not Path("tensor_names.json").is_file():
    mknet(setup_config)
with open("tensor_names.json", "r") as f:
    tensor_names = json.load(f)
    logging.warning(f"{tensor_names}")


if __name__ == "__main__":
    train(setup_config["NUM_ITERATIONS"], setup_config, tensor_names, TENSOR_NAMES)
