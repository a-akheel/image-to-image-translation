"""Train an AttGAN baseline from annotated manifests; resume without skipping work."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from .data import dataset, read_manifest
from .attributes import targets
from .infer import save_grid
from .training import Trainer


def validate_config(c):
    for key in ('batch_size','epochs','n_d','dim','fc_dim','sample_every','save_every'):
        if not isinstance(c[key],int) or isinstance(c[key],bool) or c[key]<1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0<=c['decay_start']<c['epochs']:
        raise ValueError('decay_start must be between 0 and epochs-1')
    if c['crop_size']<64 or c['crop_size']%32 or c['load_size']<c['crop_size']:
        raise ValueError('crop_size must be >=64, a multiple of 32, and <=load_size')
    if not np.isfinite(c['learning_rate']) or c['learning_rate']<=0:
        raise ValueError('learning_rate must be positive and finite')


def learning_rate(config, epoch):
    rate=config['learning_rate']
    if epoch>=config['decay_start']:
        rate*=1-(epoch-config['decay_start']+1)/(config['epochs']-config['decay_start']+1)
    return rate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/baseline.json')
    parser.add_argument('--images',required=True)
    parser.add_argument('--manifests',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-steps',type=int,help='Stop at this total batch-update count for a pilot')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text()); validate_config(config)
    if args.max_steps is not None and args.max_steps<1:
        parser.error('--max-steps must be positive')
    manifests=Path(args.manifests)
    splits={name:read_manifest(manifests/f'{name}.json',args.images) for name in ('train','val','test')}
    sets={name:{r['file'] for r in rows} for name,rows in splits.items()}
    if sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test']:
        raise ValueError('Train/validation/test images overlap')
    if len(splits['train'])<config['batch_size']:
        raise ValueError('Training split has no full batches')
    config['manifest_sha256']={name:hashlib.sha256((manifests/f'{name}.json').read_bytes()).hexdigest()
                               for name in splits}
    config['images']=str(Path(args.images).resolve())
    run=Path(args.run)
    if args.resume:
        if json.loads((run/'config.json').read_text())!=config:
            raise ValueError('Resume configuration or manifests differ from the saved run')
    else:
        run.mkdir(parents=True,exist_ok=False)
        (run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    trainer=Trainer(config['crop_size'],config['dim'],config['fc_dim'],config['learning_rate'],config['seed'])
    if args.resume:
        trainer.restore(run/'checkpoints')
    options={key:config[key] for key in ('seed','load_size','crop_size')}
    val_images,val_labels=next(iter(dataset(splits['val'],args.images,1,**options)))
    writer=tf.summary.create_file_writer(str(run/'tensorboard'))

    def save():
        trainer.save(run/'checkpoints')
        trainer.generator.save_weights(run/'generator.weights.h5')

    def sample():
        labels=val_labels.numpy()
        reconstruction=trainer.generator((val_images,labels*2-1),training=False).numpy()[0]
        edit=targets(labels,{'Eyeglasses':1-int(labels[0,6])},intensity=2)
        edited=trainer.generator((val_images,edit),training=False).numpy()[0]
        save_grid([val_images[0].numpy(),reconstruction,edited],['Validation input','Reconstruction','Toggle glasses'],
                  run/'samples'/f'{int(trainer.step):08d}.png')

    completed=False
    try:
        with (run/'metrics.jsonl').open('a') as log:
            for epoch in range(int(trainer.epoch),config['epochs']):
                rate=learning_rate(config,epoch)
                trainer.g_optimizer.learning_rate.assign(rate)
                trainer.d_optimizer.learning_rate.assign(rate)
                batches=dataset(splits['train'],args.images,config['batch_size'],epoch=epoch,training=True,**options)
                start_batch=int(trainer.batch)
                for images,labels in batches.skip(start_batch):
                    if args.max_steps is not None and int(trainer.step)>=args.max_steps:
                        sample(); return
                    start=time.perf_counter()
                    metrics=trainer.update(images,labels,config['n_d'])
                    metrics.update(step=int(trainer.step),epoch=epoch,batch=int(trainer.batch),
                                   seconds=time.perf_counter()-start,learning_rate=rate)
                    log.write(json.dumps(metrics)+'\n'); log.flush()
                    with writer.as_default():
                        for key,value in metrics.items():
                            tf.summary.scalar(key,value,step=int(trainer.step))
                    if int(trainer.step)%config['sample_every']==0:
                        sample()
                    if int(trainer.step)%config['save_every']==0:
                        save()
                    print(json.dumps(metrics),flush=True)
                trainer.epoch.assign(epoch+1); trainer.batch.assign(0)
                save()
            completed=True
            sample()
    finally:
        save(); writer.close()
        (run/'status.json').write_text(json.dumps({'completed':completed,'step':int(trainer.step),
            'epoch':int(trainer.epoch),'batch':int(trainer.batch)},indent=2)+'\n')


if __name__=='__main__':
    main()
