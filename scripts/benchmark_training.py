"""Measure full-size training mechanics on synthetic tensors, not model quality."""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
import tensorflow as tf
from attgan.training import Trainer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--output',default='artifacts/training-benchmark.json')
    args=p.parse_args()
    if min(args.batch_size,args.repetitions)<1:
        p.error('Batch size and repetitions must be positive')
    t=Trainer()
    rng=np.random.default_rng(100)
    images=tf.constant(rng.uniform(-1,1,(args.batch_size,128,128,3)),tf.float32)
    labels=tf.constant(rng.integers(0,2,(args.batch_size,13)),tf.float32)
    times={}
    for name,step in [('discriminator',t.d_step),('generator',t.g_step)]:
        step(images,labels)  # Compile and warm up, excluded from timing.
        samples=[]
        for _ in range(args.repetitions):
            start=time.perf_counter()
            result={k:float(v) for k,v in step(images,labels).items()}
            samples.append(time.perf_counter()-start)
        times[name]={'median_seconds':float(np.median(samples)),'runs_seconds':samples,
                     'last_losses':result}
    report={'scope':'Full architecture, synthetic random images, CPU/GPU mechanics only. Not quality or a full-training ETA.',
        'platform':platform.platform(),'tensorflow':tf.__version__,
        'devices':[d.device_type for d in tf.config.list_physical_devices()],
        'batch_size':args.batch_size,'crop_size':128,'repetitions':args.repetitions,
        'generator_parameters':t.generator.count_params(),'discriminator_parameters':t.discriminator.count_params(),
        'timings':times}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
