"""Use an upstream example input to demonstrate newly executed inference.

The conditioning vector is an explicit illustrative setting, not verified source
annotations. Do not use this fixture to evaluate reconstruction/attribute accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from attgan.attributes import targets, ATTRIBUTES
from attgan.reference import import_generator
from attgan.infer import save_grid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--upstream-grid", required=True)
    p.add_argument("--output", default="artifacts/demo.png")
    args=p.parse_args()
    # First input cell from upstream's published 128px comparison sheet.
    with Image.open(args.upstream_grid) as sheet:
        x=np.asarray(sheet.convert("RGB").crop((0,0,128,128)),dtype=np.float32)[None]/127.5-1
    source=np.zeros((1,13),np.float32)
    # This is a requested base condition, not labels inferred from the person.
    source[0,ATTRIBUTES.index("No_Beard")]=1
    model=import_generator(args.graph)
    variants=[("Base condition", {}), ("Glasses", {"Eyeglasses":1}),
              ("Bangs", {"Bangs":1}), ("Blond hair", {"Blond_Hair":1}),
              ("Blond + glasses", {"Blond_Hair":1,"Eyeglasses":1})]
    images=[x[0]]
    for label, edits in variants:
        images.append(model((x,targets(source,edits,2))).numpy()[0])
    save_grid(images,["Input"]+[v[0] for v in variants],args.output)
    # Retain the exact input for the separate numerical-parity command.
    from attgan.infer import image_array
    Image.fromarray(image_array(x[0])).save(Path(args.output).with_name('demo-input.png'))
    Path(args.output).with_suffix(".json").write_text(json.dumps({
        "source": "LynnHo/AttGAN-Tensorflow pics/sample_validation.jpg, first 128x128 input cell",
        "upstream_revision": "f1231fcc581133c9a7d8451e26a8bcd293fe03d5",
        "source_grid_sha256": hashlib.sha256(Path(args.upstream_grid).read_bytes()).hexdigest(),
        "model": "Authors' pretrained AttGAN_128, imported into Keras",
        "own_training": False, "source_annotations_available": False,
        "illustrative_base_labels": source[0].tolist(), "attribute_order": ATTRIBUTES,
        "variants": variants, "intensity":2,
        "preprocessing": "Already preprocessed upstream input cell; no additional resize/crop",
        "limitations": "Demonstration only; not held-out evaluation, reconstruction accuracy, or our training results."
    },indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
