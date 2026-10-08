"""Run the authors' frozen graph and transfer its constants to Keras.

No random fallback: missing or incompatible weights must fail visibly.
"""
from pathlib import Path
import hashlib
import numpy as np
import tensorflow as tf
from .models import Generator

GENERATOR_SHA256 = "e595d1602e9ff3267c138342c6dfb530ef65842f1452d1ca6322760f0208a60e"


def read_graph(path, verify=True):
    contents = Path(path).read_bytes()
    if verify and hashlib.sha256(contents).hexdigest() != GENERATOR_SHA256:
        raise ValueError("This is not the verified upstream AttGAN_128 generator.pb.")
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(contents)
    return graph_def


class FrozenGenerator:
    def __init__(self, path):
        graph_def = read_graph(path)
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")
        self.images = self.graph.get_tensor_by_name("xa:0")
        self.attributes = self.graph.get_tensor_by_name("b_:0")
        self.output = self.graph.get_tensor_by_name("xb:0")
        self.session = tf.compat.v1.Session(graph=self.graph)

    def __call__(self, inputs):
        images, attributes = inputs
        return self.session.run(self.output, {self.images: images, self.attributes: attributes})

    def close(self):
        self.session.close()


def import_generator(path):
    graph = read_graph(path)
    constants = {n.name: tf.make_ndarray(n.attr["value"].tensor)
                 for n in graph.node if n.op == "Const"}
    model = Generator()
    model((tf.zeros([1, 128, 128, 3]), tf.zeros([1, 13])))
    used = []
    for scope, blocks, prefix in [("UNetGenc", model.encoder, "Conv"),
                                   ("UNetGdec", model.decoder, "Conv2d_transpose")]:
        for i, block in enumerate(blocks):
            name = f"{scope}/{prefix}" + (f"_{i}" if i else "")
            key = name + "/weights"
            block.conv.set_weights([constants[key]])
            used.append(key)
            keys = [name + "/BatchNorm/" + n for n in
                    ("gamma", "beta", "moving_mean", "moving_variance")]
            block.norm.set_weights([constants[k] for k in keys])
            used.extend(keys)
    keys = ["UNetGdec/Conv2d_transpose_4/" + name for name in ("weights", "biases")]
    model.output_layer.set_weights([constants[k] for k in keys])
    used.extend(keys)
    if len(used) != len(model.weights):
        raise ValueError("Not every Keras weight was populated from the reference graph.")
    for weight in model.weights:
        if not np.isfinite(weight.numpy()).all():
            raise ValueError("Non-finite pretrained weight.")
    return model
