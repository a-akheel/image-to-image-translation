"""Strict CelebA annotation parsing and reproducible, disjoint file splits."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from .attributes import ATTRIBUTES, binary_labels


def image_path(root, name):
    root=Path(root).resolve()
    path=(root/name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes dataset directory: {name}")
    if not path.is_file():
        raise ValueError(f"Missing image: {name}")
    return path


def parse_annotations(path, image_dir):
    lines=Path(path).read_text().splitlines()
    if len(lines)<3:
        raise ValueError("Expected count, attribute header, then annotation rows.")
    count=int(lines[0].strip())
    header=lines[1].split()
    if len(set(header))!=len(header) or not set(ATTRIBUTES).issubset(header):
        raise ValueError("Missing or duplicate attribute names in annotation header.")
    records=[]; seen=set()
    for line in lines[2:]:
        fields=line.split()
        if not fields:
            continue
        if len(fields)!=len(header)+1:
            raise ValueError("Annotation row length does not match header.")
        name=fields[0]
        if name in seen:
            raise ValueError(f"Duplicate image: {name}")
        image_path(image_dir, name)
        values=[int(x) for x in fields[1:]]
        if any(x not in (-1,1) for x in values):
            raise ValueError("Raw annotation values must be -1 or 1.")
        labels=[(values[header.index(a)]+1)//2 for a in ATTRIBUTES]
        records.append({"file":name,"labels":labels}); seen.add(name)
    if count!=len(records):
        raise ValueError(f"Annotation count {count} does not match {len(records)} rows.")
    return records


def prepare(annotation_path, image_dir, output, train_count=26500, val_count=500, seed=100):
    records=parse_annotations(annotation_path,image_dir)
    if min(train_count,val_count)<1 or train_count+val_count>=len(records):
        raise ValueError("All three splits must be nonempty.")
    random.Random(seed).shuffle(records)
    splits={"train":records[:train_count],"val":records[train_count:train_count+val_count],
            "test":records[train_count+val_count:]}
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    for name, rows in splits.items():
        (output/f"{name}.json").write_text(json.dumps({"attributes":ATTRIBUTES,"records":rows},indent=2)+"\n")
    info={"seed":seed,"counts":{k:len(v) for k,v in splits.items()},
          "annotation_sha256":hashlib.sha256(Path(annotation_path).read_bytes()).hexdigest(),
          "split_policy":"Seeded image-disjoint split following upstream HQ counts; not identity-disjoint."}
    (output/"metadata.json").write_text(json.dumps(info,indent=2)+"\n")
    return info


def read_manifest(path, image_dir):
    obj=json.loads(Path(path).read_text())
    if obj["attributes"]!=list(ATTRIBUTES):
        raise ValueError("Manifest attribute order is incompatible with this model.")
    records=obj["records"]
    if not records or len({r['file'] for r in records})!=len(records):
        raise ValueError("Empty manifest or duplicate image names.")
    for row in records:
        image_path(image_dir,row['file']); binary_labels(row['labels'])
    return records


def dataset(records, image_dir, batch_size, epoch=0, seed=100, load_size=128,
            crop_size=128, training=False):
    if batch_size<1 or load_size<crop_size or crop_size%32 or crop_size<64:
        raise ValueError("Invalid batch size or image dimensions (crop >=64, multiple of 32).")
    if training and len(records)<batch_size:
        raise ValueError("Training split is smaller than one batch.")
    order=np.arange(len(records))
    if training:
        np.random.default_rng(seed+epoch).shuffle(order)
    paths=[str(image_path(image_dir,records[i]['file'])) for i in order]
    labels=np.asarray([records[i]['labels'] for i in order],np.float32)
    ds=tf.data.Dataset.from_tensor_slices((paths,labels,np.arange(len(order),dtype=np.int32)))
    def decode(path, label, index):
        x=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False)
        x.set_shape([None,None,3])
        x=tf.image.resize(x,[load_size,load_size])
        if training:
            key=tf.stack([tf.cast(seed+epoch,tf.int32),index])
            keys=tf.random.experimental.stateless_split(key,2)
            x=tf.image.stateless_random_flip_left_right(x,keys[0])
            x=tf.image.stateless_random_crop(x,[crop_size,crop_size,3],keys[1])
        else:
            start=(load_size-crop_size)//2
            x=x[start:start+crop_size,start:start+crop_size]
        return tf.clip_by_value(x,0,255)/127.5-1,label
    return ds.map(decode,num_parallel_calls=2,deterministic=True).batch(
        batch_size,drop_remainder=training).prefetch(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--images',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--train-count',type=int,default=26500)
    parser.add_argument('--val-count',type=int,default=500)
    parser.add_argument('--seed',type=int,default=100)
    args=parser.parse_args()
    print(json.dumps(prepare(args.annotations,args.images,args.output,args.train_count,args.val_count,args.seed),indent=2))


if __name__=='__main__':
    main()
