# Specification decisions

Page numbers are printed report pages (PDF page index is +8).

| Requirement | Decision |
| --- | --- |
| Facial editing, pp. 1, 6, 30-34 | Implement the 13-attribute editor, not street-scene/animal translation from the old README. |
| VAE terminology, pp. 2-4, 20-21 | Follow Figure 12 and appendix pp. 44-51: deterministic encoder/decoder, no KL term or sampled posterior. |
| Unpaired images, pp. 1-2 | Attribute annotations are still required for classification losses. |
| Networks, pp. 31-32 | Five stages, dim 64, one shortcut and attribute reinjection; normalization and activations follow upstream. |
| Losses, appendix pp. 49-51 | WGAN-GP weight 10; real classification 1; generated classification 10; reconstruction L1 100. |
| Preprocessing, pp. 19, 29, 41-44 | HQ baseline uses aligned HQ images. Arbitrary-photo alignment remains unimplemented. Pretrained weights use the original 143-to-128 crop. |
| CelebA sample versus HQ, pp. 28, 33, 36, 48 | Select appendix's HQ path at 128px; do not claim exact reproduction of the sampled-CelebA experiment. |
| 90/10 versus upstream HQ split | Use upstream seed 100 and 26,500/500/3,000 split. Image-disjoint, not guaranteed identity-disjoint. |
| 100 epochs in results versus 60 in appendix | Select 60, decay from 30; other schedules need separately declared experiments. |
| Save/sample counts | Count batch updates. Sample every 600; save every 6,000 plus epoch/exit saves. Five D updates per G update. |
| Noise-created fakes in Figure 10 | Use generated images, consistent with Figure 12 and training code. |
| Single/multiple edits, pp. 34-36 | Explicit target values, conflicts checked, intensity, comparison grids. |
| Accuracy/SSIM claims, pp. 36-37 | Do not reuse unverified accuracy claims. Reconstruction metrics implemented; independent attribute evaluation outstanding. |

## Training differences

- Keras and GradientTape replace TF1 sessions and tf.contrib.
- Adam epsilon 1e-8; BN epsilon 0.001 and momentum 0.999; instance-norm epsilon 1e-6;
  LeakyReLU slope 0.2, matching the reference.
- Add 1e-12 under the gradient-penalty square root to keep derivatives finite at zero.
  This is a small intentional numerical change.
- Checkpointed RNG and stateless epoch/index augmentation replace upstream randomness.
- As upstream, generator BN updates during D steps. Reconstruction and editing share
  a single encoder pass during G steps.
- Inference equivalence to frozen operations was checked in TF2. Exact TF1 training
  trajectories and training-time normalization equivalence have not been established.

## Evidence and outstanding work

`reference-parity.json` records six inference comparisons and warm CPU latency.
`training-benchmark.json` records synthetic full-model mechanics, not quality.
Unit tests cover gradient flow, optimizer separation, constraints, manifests, and
checkpoint/next-update agreement. The CLI smoke run completed twelve updates over
three epochs, stopping at six and resuming at seven, then exported/evaluated weights.

Real-data training, independent attribute metrics, arbitrary-photo alignment, production
deployment, and research extensions remain incomplete. Dataset download hit the
publisher's quota; its README also directs attribute-label requests to the CelebA team.
