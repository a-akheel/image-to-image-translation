"""Compare frozen TF1 graph operations with the weight-mapped Keras generator."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from attgan.reference import FrozenGenerator, import_generator
from attgan.infer import read_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--image", help="Optional pre-aligned 128px input for image parity")
    p.add_argument("--report", default="artifacts/parity.json")
    args = p.parse_args()
    tf.keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)
    frozen = FrozenGenerator(args.graph)
    keras = import_generator(args.graph)
    cases = [("random_batch", rng.uniform(-1, 1, (2,128,128,3)).astype(np.float32)),
             ("zeros", np.zeros((1,128,128,3), np.float32))]
    if args.image:
        cases.append(("real_image", read_image(args.image, 128)[None]))
    results = []
    try:
        for name, images in cases:
            for intensity in (1.0, 2.0):
                attributes = (rng.integers(0,2,(len(images),13))*2-1).astype(np.float32)*intensity
                expected = frozen((images, attributes))
                actual = keras((images, attributes), training=False).numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
                results.append({"case": name, "intensity": intensity,
                                "max_absolute_error": float(np.max(abs(actual-expected))),
                                "mean_absolute_error": float(np.mean(abs(actual-expected)))})
        # Warm inference only, batch 1. Excludes model load and preprocessing.
        x = cases[0][1][:1]; a = np.ones((1,13), np.float32)
        keras((x,a)).numpy()
        times=[]
        for _ in range(10):
            start=time.perf_counter(); keras((x,a)).numpy()
            times.append((time.perf_counter()-start)*1000)
    finally:
        frozen.close()
    report = {"tensorflow": tf.__version__, "python": platform.python_version(),
              "platform": platform.platform(), "devices": [d.device_type for d in tf.config.list_physical_devices()],
              "generator_sha256": hashlib.sha256(Path(args.graph).read_bytes()).hexdigest(),
              "image_sha256": hashlib.sha256(Path(args.image).read_bytes()).hexdigest() if args.image else None,
              "generator_parameters": keras.count_params(), "cases": results,
              "atol": 1e-4, "rtol": 1e-4, "passed": True,
              "warm_batch1_runs": 10, "warm_batch1_median_ms": float(np.median(times)),
              "scope": "Inference equivalence to the frozen upstream graph; not training parity or editing quality."}
    out=Path(args.report); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
