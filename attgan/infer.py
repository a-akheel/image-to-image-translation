"""Inference on an already aligned face; no hidden face detection or label guessing."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
from .attributes import ATTRIBUTES, binary_labels, targets
from .models import Generator
from .reference import FrozenGenerator, import_generator


def read_image(path, load_size=143, crop_size=128):
    if load_size < crop_size:
        raise ValueError("load_size must be at least crop_size")
    with Image.open(path) as image:
        array = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    # Match the authors' TF resize and integer center crop, not PIL's bicubic.
    image = tf.image.resize(array, [load_size, load_size], method="bilinear")
    start = (load_size-crop_size)//2
    return (image[start:start+crop_size, start:start+crop_size] / 127.5 - 1).numpy()


def image_array(x):
    return np.rint((np.clip(x, -1, 1) + 1) * 127.5).astype(np.uint8)


def save_grid(images, labels, path):
    size = images[0].shape[0]
    width = max(size, 160)
    canvas = Image.new("RGB", (len(images)*width, size+42), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels, strict=True)):
        canvas.paste(Image.fromarray(image_array(image)), (i*width+(width-size)//2, 36))
        draw.text((i*width+5, 10), label, fill="black")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", required=True, help="13 comma-separated 0/1 source labels")
    parser.add_argument("--set", nargs="+", required=True, metavar="ATTRIBUTE=0|1")
    parser.add_argument("--intensity", type=float, default=1)
    parser.add_argument("--load-size", type=int, default=143)
    parser.add_argument("--backend", choices=["keras", "frozen"], default="keras")
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--graph", help="Verified authors' generator.pb")
    weights.add_argument("--weights", help="Keras .weights.h5 from a full-size run")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        source = binary_labels([int(x) for x in args.labels.split(",")])[None]
        edits = {}
        for spec in args.set:
            name, value = spec.split("=", 1)
            if name in edits:
                raise ValueError(f"Duplicate edit: {name}")
            edits[name] = int(value)
        conditioning = targets(source, edits, args.intensity)
        if args.backend == "frozen" and args.weights:
            raise ValueError("The frozen backend requires --graph.")
    except ValueError as error:
        parser.error(str(error))
    x = read_image(args.image, args.load_size)[None]
    if args.graph:
        model = FrozenGenerator(args.graph) if args.backend == "frozen" else import_generator(args.graph)
    else:
        model = Generator()
        model((tf.zeros_like(x), tf.zeros([1, 13])))
        model.load_weights(args.weights)
    try:
        reconstruction = np.asarray(model((x, source*2-1)))[0]
        edited = np.asarray(model((x, conditioning)))[0]
    finally:
        if isinstance(model, FrozenGenerator):
            model.close()
    save_grid([x[0], reconstruction, edited], ["Input", "Reconstruction", "Requested edits"], args.output)
    metadata = {"attributes": ATTRIBUTES, "source_labels": source[0].tolist(),
                "edits": edits, "conditioning": conditioning[0].tolist(),
                "backend": args.backend, "weights": args.graph or args.weights,
                "pretrained": bool(args.graph), "input": args.image,
                "load_size": args.load_size, "crop_size": 128}
    Path(args.output).with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
