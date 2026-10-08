"""Reconstruction evaluation only; SSIM does not establish editing success."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from .data import dataset,read_manifest
from .models import Generator
from .reference import import_generator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',required=True)
    parser.add_argument('--manifest',required=True)
    weights=parser.add_mutually_exclusive_group(required=True)
    weights.add_argument('--graph')
    weights.add_argument('--weights')
    parser.add_argument('--config',default='configs/baseline.json',help='Architecture and preprocessing for own weights')
    parser.add_argument('--load-size',type=int,default=143,help='Preprocessing size for upstream graph')
    parser.add_argument('--limit',type=int,default=100)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.limit<1:
        parser.error('--limit must be positive')
    records=read_manifest(args.manifest,args.images)[:args.limit]
    if args.graph:
        model=import_generator(args.graph); crop_size=128; load_size=args.load_size
    else:
        c=json.loads(Path(args.config).read_text())
        crop_size=c['crop_size']; load_size=c['load_size']
        model=Generator(dim=c['dim'])
        model((tf.zeros([1,crop_size,crop_size,3]),tf.zeros([1,13])))
        model.load_weights(args.weights)
    scores=[]
    for i,(images,labels) in enumerate(dataset(records,args.images,1,load_size=load_size,crop_size=crop_size)):
        reconstruction=model((images,labels*2-1),training=False)
        score={'file':records[i]['file'],
               'ssim':float(tf.image.ssim((images+1)/2,(reconstruction+1)/2,max_val=1)[0]),
               'l1_0_to_1':float(tf.reduce_mean(tf.abs(images-reconstruction))/2)}
        scores.append(score)
    report={'scope':'Reconstruction only, not attribute editing accuracy or identity preservation',
            'attribute_edit_accuracy':None,'unedited_attribute_preservation':None,
            'missing_evaluation':'Requires an independent calibrated attribute evaluator and a held-out labeled dataset',
            'count':len(scores),'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            'model_sha256':hashlib.sha256(Path(args.graph or args.weights).read_bytes()).hexdigest(),
            'load_size':load_size,'crop_size':crop_size,'tensorflow':tf.__version__,
            'mean_ssim':float(np.mean([r['ssim'] for r in scores])),
            'mean_l1_0_to_1':float(np.mean([r['l1_0_to_1'] for r in scores])), 'per_image':scores}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_image'},indent=2))


if __name__=='__main__':
    main()
