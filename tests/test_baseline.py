import json
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image
from attgan.attributes import ATTRIBUTES,targets
from attgan.data import prepare,read_manifest,dataset
from attgan.models import Generator
from attgan.training import Trainer
from attgan.train import validate_config,learning_rate


def test_attribute_edit_preserves_input_and_resolves_hair():
    source=np.zeros((2,13),np.float32); source[:,2]=1
    original=source.copy()
    edited=targets(source,{'Blond_Hair':1,'Eyeglasses':1},2)
    np.testing.assert_array_equal(source,original)
    np.testing.assert_array_equal(edited[:,[2,3,6]],[[-1,2,2],[-1,2,2]])
    np.testing.assert_array_equal(edited[:,7:],source[:,7:]*2-1)


@pytest.mark.parametrize('edits',[{'Bangs':1,'Bald':1},{'Black_Hair':1,'Blond_Hair':1},
                                  {'unknown':1},{'Eyeglasses':2}])
def test_invalid_edits(edits):
    with pytest.raises(ValueError): targets(np.zeros(13),edits)


@pytest.mark.parametrize('intensity',[0,-1,4,float('nan'),float('inf')])
def test_invalid_intensity(intensity):
    with pytest.raises(ValueError): targets(np.zeros(13),{'Bangs':1},intensity)


def test_bad_label_count_and_nonbinary_labels():
    for labels in [np.zeros(12),np.full(13,-1),np.full(13,np.nan)]:
        with pytest.raises(ValueError): targets(labels,{})


@pytest.fixture
def annotations(tmp_path):
    images=tmp_path/'images'; images.mkdir()
    rng=np.random.default_rng(7)
    rows=['12',' '.join(reversed(ATTRIBUTES))]
    for i in range(12):
        Image.fromarray(rng.integers(0,256,(72,72,3),dtype=np.uint8)).save(images/f'{i}.png')
        rows.append(f'{i}.png '+' '.join(['-1']*12+['1']))
    path=tmp_path/'attributes.txt'; path.write_text('\n'.join(rows))
    return path,images


def test_manifest_order_split_and_determinism(annotations,tmp_path):
    path,images=annotations
    prepare(path,images,tmp_path/'a',8,2)
    prepare(path,images,tmp_path/'b',8,2)
    groups=[]
    for name in ('train','val','test'):
        assert (tmp_path/'a'/f'{name}.json').read_bytes()==(tmp_path/'b'/f'{name}.json').read_bytes()
        records=read_manifest(tmp_path/'a'/f'{name}.json',images)
        assert all(row['labels']==[1]+[0]*12 for row in records)
        groups.append({r['file'] for r in records})
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert len(set.union(*groups))==12


def test_dataset_resume_order_and_augmentation(annotations,tmp_path):
    path,images=annotations; prepare(path,images,tmp_path/'split',8,2)
    rows=read_manifest(tmp_path/'split/train.json',images)
    options=dict(batch_size=2,epoch=3,seed=100,load_size=72,crop_size=64,training=True)
    full=list(dataset(rows,images,**options).as_numpy_iterator())
    resumed=list(dataset(rows,images,**options).skip(2).as_numpy_iterator())
    for expected,actual in zip(full[2:],resumed,strict=True):
        np.testing.assert_array_equal(expected[0],actual[0])
        np.testing.assert_array_equal(expected[1],actual[1])
    assert full[0][0].shape==(2,64,64,3)
    assert np.max(full[0][0])<=1 and np.min(full[0][0])>=-1


@pytest.mark.parametrize('failure',['count','duplicate','invalid_label','missing','traversal'])
def test_invalid_annotation_rejected(annotations,tmp_path,failure):
    path,images=annotations; lines=path.read_text().splitlines()
    if failure=='count': lines[0]='13'
    if failure=='duplicate': lines[-1]=lines[-2]
    if failure=='invalid_label': lines[2]=lines[2].replace('-1','0')
    if failure=='missing': lines[2]=lines[2].replace('0.png','absent.png')
    if failure=='traversal': lines[2]=lines[2].replace('0.png','../0.png')
    path.write_text('\n'.join(lines))
    with pytest.raises(ValueError): prepare(path,images,tmp_path/'out',8,2)


def test_generator_range_shape_and_conditioning():
    tf.keras.utils.set_random_seed(13)
    model=Generator(dim=4)
    x=tf.random.uniform([2,64,64,3],-1,1)
    a=model((x,tf.ones([2,13])))
    b=model((x,-tf.ones([2,13])))
    assert a.shape==x.shape
    assert float(tf.reduce_max(tf.abs(a)))<=1
    assert not np.allclose(a,b)


@pytest.fixture(scope='module')
def training_batch():
    rng=np.random.default_rng(50)
    return (tf.constant(rng.uniform(-1,1,(2,64,64,3)),tf.float32),
            tf.constant(rng.integers(0,2,(2,13)),tf.float32))


def snapshot(variables): return [v.numpy().copy() for v in variables]


def test_discriminator_and_generator_update_separation(training_batch):
    t=Trainer(crop_size=64,dim=4,fc_dim=16)
    g_before=snapshot(t.generator.trainable_variables)
    d_before=snapshot(t.discriminator.trainable_variables)
    d=t.d_step(*training_batch)
    for before,after in zip(g_before,t.generator.trainable_variables,strict=True):
        np.testing.assert_array_equal(before,after.numpy())
    assert any(not np.array_equal(a,b.numpy()) for a,b in zip(d_before,t.discriminator.trainable_variables))
    assert float(d['d_total'])==pytest.approx(float(d['d_adversarial']+10*d['gradient_penalty']+d['d_attribute']),rel=1e-5)
    d_before=snapshot(t.discriminator.variables)
    g=t.g_step(*training_batch)
    assert float(g['g_total'])==pytest.approx(float(g['g_adversarial']+10*g['g_attribute']+100*g['g_reconstruction']),rel=1e-5)
    for before,after in zip(d_before,t.discriminator.variables,strict=True):
        np.testing.assert_array_equal(before,after.numpy())
    assert any(not np.array_equal(a,b.numpy()) for a,b in zip(g_before,t.generator.trainable_variables))


def test_checkpoint_resume_matches_next_update(training_batch,tmp_path):
    t=Trainer(crop_size=64,dim=4,fc_dim=16)
    # Reach just before a generator update to exercise scheduling and both optimizers.
    for _ in range(5): t.update(*training_batch)
    t.save(tmp_path/'checkpoints')
    restored=Trainer(crop_size=64,dim=4,fc_dim=16,seed=999)
    restored.restore(tmp_path/'checkpoints')
    assert int(restored.step)==int(restored.batch)==5
    for variables_a,variables_b in [(t.generator.variables,restored.generator.variables),
        (t.discriminator.variables,restored.discriminator.variables),
        (t.d_optimizer.variables,restored.d_optimizer.variables),
        (t.g_optimizer.variables,restored.g_optimizer.variables)]:
        for a,b in zip(variables_a,variables_b,strict=True):
            np.testing.assert_array_equal(a.numpy(),b.numpy())
    for _ in range(2):
        expected=t.update(*training_batch); actual=restored.update(*training_batch)
        assert actual==pytest.approx(expected,rel=1e-5,abs=1e-5)
    assert int(restored.g_optimizer.iterations)==1
    assert int(restored.d_optimizer.iterations)==6
    for a,b in zip(t.generator.variables,restored.generator.variables,strict=True):
        np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=1e-5,atol=1e-5)


def test_missing_checkpoint_does_not_start_fresh(tmp_path):
    t=Trainer(crop_size=64,dim=4,fc_dim=16)
    with pytest.raises(FileNotFoundError): t.restore(tmp_path/'missing')


def test_config_and_schedule():
    c=json.loads(Path('configs/baseline.json').read_text())
    validate_config(c)
    assert learning_rate(c,0)==c['learning_rate']
    assert learning_rate(c,59)<learning_rate(c,30)<learning_rate(c,29)
    c['batch_size']=0
    with pytest.raises(ValueError): validate_config(c)
