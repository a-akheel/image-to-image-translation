"""Generate synthetic plumbing fixtures, NEVER a face dataset or quality benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from attgan.attributes import ATTRIBUTES
from attgan.data import prepare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='artifacts/synthetic-smoke')
    args=parser.parse_args(); root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    images=root/'images'; images.mkdir()
    rng=np.random.default_rng(100)
    rows=['12',' '.join(ATTRIBUTES)]
    for i in range(12):
        Image.fromarray(rng.integers(0,256,(72,72,3),dtype=np.uint8)).save(images/f'{i}.png')
        labels=rng.choice([-1,1],13)
        rows.append(f'{i}.png '+' '.join(map(str,labels)))
    annotation=root/'annotations.txt'; annotation.write_text('\n'.join(rows)+'\n')
    prepare(annotation,images,root/'manifests',8,2)
    c=json.loads(Path('configs/baseline.json').read_text())
    c.update(crop_size=64,load_size=72,dim=4,fc_dim=16,batch_size=2,epochs=3,
             decay_start=1,sample_every=6,save_every=6)
    (root/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    print(root)


if __name__=='__main__':
    main()
