"""WGAN-GP + attribute BCE + L1 reconstruction; no proposed extension yet."""
from pathlib import Path
import tensorflow as tf
from .models import Generator, Discriminator


def bce(labels, logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))


class Trainer:
    def __init__(self, crop_size=128, dim=64, fc_dim=1024, learning_rate=2e-4, seed=100):
        tf.keras.utils.set_random_seed(seed)
        self.generator=Generator(dim=dim)
        self.discriminator=Discriminator(dim=dim,fc_dim=fc_dim)
        x=tf.zeros([1,crop_size,crop_size,3]); a=tf.zeros([1,13])
        self.generator((x,a)); self.discriminator(x)
        self.g_optimizer=tf.keras.optimizers.Adam(learning_rate,beta_1=0.5,beta_2=0.999,epsilon=1e-8)
        self.d_optimizer=tf.keras.optimizers.Adam(learning_rate,beta_1=0.5,beta_2=0.999,epsilon=1e-8)
        self.g_optimizer.build(self.generator.trainable_variables)
        self.d_optimizer.build(self.discriminator.trainable_variables)
        self.rng=tf.random.Generator.from_seed(seed)
        self.step=tf.Variable(0,dtype=tf.int64,trainable=False)
        self.epoch=tf.Variable(0,dtype=tf.int64,trainable=False)
        self.batch=tf.Variable(0,dtype=tf.int64,trainable=False)
        self.checkpoint=tf.train.Checkpoint(generator=self.generator,discriminator=self.discriminator,
            g_optimizer=self.g_optimizer,d_optimizer=self.d_optimizer,rng=self.rng,
            step=self.step,epoch=self.epoch,batch=self.batch)

    def shuffled(self, labels):
        return tf.gather(labels,tf.argsort(self.rng.uniform([tf.shape(labels)[0]])))

    def apply(self, tape, loss, variables, optimizer):
        gradients=tape.gradient(loss,variables)
        tf.debugging.assert_all_finite(loss,"Nonfinite loss")
        if any(g is None for g in gradients):
            raise ValueError("A trainable weight is disconnected from its objective.")
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient,"Nonfinite gradient")
        optimizer.apply_gradients(zip(gradients,variables))

    @tf.function(reduce_retracing=True)
    def d_step(self, images, labels):
        target=self.shuffled(labels)
        # Upstream updates generator BN statistics during D steps too.
        fake=tf.stop_gradient(self.generator((images,target*2-1),training=True))
        with tf.GradientTape() as tape:
            real_score,real_attributes=self.discriminator(images,training=True)
            fake_score,_=self.discriminator(fake,training=True)
            alpha=self.rng.uniform([tf.shape(images)[0],1,1,1])
            interpolated=images+alpha*(fake-images)
            with tf.GradientTape() as penalty_tape:
                penalty_tape.watch(interpolated)
                interpolated_score,_=self.discriminator(interpolated,training=True)
            gradient=penalty_tape.gradient(interpolated_score,interpolated)
            norm=tf.sqrt(tf.reduce_sum(tf.square(gradient),axis=[1,2,3])+1e-12)
            penalty=tf.reduce_mean(tf.square(norm-1))
            adversarial=tf.reduce_mean(fake_score)-tf.reduce_mean(real_score)
            classification=bce(labels,real_attributes)
            loss=adversarial+10*penalty+classification
        self.apply(tape,loss,self.discriminator.trainable_variables,self.d_optimizer)
        return {"d_total":loss,"d_adversarial":adversarial,"gradient_penalty":penalty,
                "d_attribute":classification}

    @tf.function(reduce_retracing=True)
    def g_step(self, images, labels):
        target=self.shuffled(labels)
        with tf.GradientTape() as tape:
            features=self.generator.encode(images,training=True)
            reconstruction=self.generator.decode(features,labels*2-1,training=True)
            fake=self.generator.decode(features,target*2-1,training=True)
            score,attributes=self.discriminator(fake,training=True)
            adversarial=-tf.reduce_mean(score)
            classification=bce(target,attributes)
            reconstruction_loss=tf.reduce_mean(tf.abs(images-reconstruction))
            loss=adversarial+10*classification+100*reconstruction_loss
        self.apply(tape,loss,self.generator.trainable_variables,self.g_optimizer)
        return {"g_total":loss,"g_adversarial":adversarial,"g_attribute":classification,
                "g_reconstruction":reconstruction_loss}

    def update(self, images, labels, n_d=5):
        # Exactly n_d D batch updates followed by one G batch update.
        if n_d<1:
            raise ValueError("n_d must be positive")
        result=(self.g_step if (int(self.step)+1)%(n_d+1)==0 else self.d_step)(images,labels)
        self.step.assign_add(1); self.batch.assign_add(1)
        return {k:float(v) for k,v in result.items()}

    def save(self, directory):
        Path(directory).mkdir(parents=True,exist_ok=True)
        manager=tf.train.CheckpointManager(self.checkpoint,str(directory),max_to_keep=3)
        return manager.save(checkpoint_number=int(self.step))

    def restore(self, directory):
        latest=tf.train.latest_checkpoint(str(directory))
        if latest is None:
            raise FileNotFoundError(f"No checkpoint in {directory}")
        # Create save_counter before strict matching; save() creates it lazily.
        _=self.checkpoint.save_counter
        self.checkpoint.restore(latest).assert_consumed()
        return latest
