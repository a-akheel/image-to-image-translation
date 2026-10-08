# Sources and attribution

- Method: Zhenliang He, Wangmeng Zuo, Meina Kan, Shiguang Shan, Xilin Chen,
  **AttGAN: Facial Attribute Editing by Only Changing What You Want**, IEEE TIP 2019,
  DOI [10.1109/TIP.2019.2916751](https://doi.org/10.1109/TIP.2019.2916751).
- Code: [LynnHo/AttGAN-Tensorflow](https://github.com/LynnHo/AttGAN-Tensorflow),
  commit `f1231fcc581133c9a7d8451e26a8bcd293fe03d5`, MIT, copyright 2018 Zhenliang He.
  Architecture, losses, preprocessing and attribute order are adapted from this source.
  Its MIT notice is retained in `LICENSE`.
- User fork: `a-akheel/AttGAN-Tensorflow`, identical to upstream at inspection. It was
  not modified. Work is in `image-to-image-translation`.
- Weights: authors' `AttGAN_128_generator_only.zip`, linked from their README,
  [download](https://drive.google.com/file/d/1lcQ-ijNrGD4919eJ5Dv-7ja5rsx5p0Tp/view).
  These were not trained by this project. Checksums are recorded in setup and reports.
- Demo input: first input cell in upstream `pics/sample_validation.jpg` at the pinned
  revision. Local generated outputs and illustrative conditioning are documented in
  `docs/demo-metadata.json`. No identity or verified source attributes are asserted.
- Specification: user-provided 2020-2021 CBIT report, **Unsupervised Image to Image
  Translation Using GAN**. Its results are treated as unverified document claims; the
  user explained the implementation was unfinished. The private PDF is not redistributed.
- Data: [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) and
  [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Obtain data/labels from
  the publishers and follow their terms. The code license does not grant rights to
  redistribute datasets, face images, or weights. Large assets are ignored by Git.

Modernization and validation used Codex assistance in October 2026. No authorship of
AttGAN, new architecture, or state-of-the-art performance is claimed.
