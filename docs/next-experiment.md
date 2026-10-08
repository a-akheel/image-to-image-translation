# One improvement after a measured baseline

Question: can we reduce unwanted changes while keeping the requested edit successful?

Candidate: add a masked preservation penalty outside the intended edit region, using
aligned face-parsing masks. This is a proposal, not an implemented or novel method.
A literature check and a measured baseline are prerequisites.

1. Acquire labeled data, verify splits, and complete a real-data pilot.
2. Validate an independent attribute classifier on held-out real images. Measure target
   attribute success and unintended attribute changes separately.
3. Start with localized edits (hair/glasses). Define regions with space for additions;
   source masks alone may prevent new bangs. Avoid naive masking of global attributes.
4. Compare baseline and extension with identical data, seeds, resolution, schedule, and
   budget. Select a small declared set of penalty weights using validation data only.
   Include a zero-penalty control. Keep test data untouched until final evaluation.
5. Compare edit success, non-edited attributes, outside-region changes, visible artifacts,
   and speed. Include failures. Higher SSIM is not success if edits simply get weaker.
6. Use multiple seeds/confidence intervals when compute permits; otherwise label results
   exploratory. Reject the extension if it reduces edit success to preserve pixels.

Attention is a separate candidate. Do not add both changes simultaneously: we need to
know which change caused any improvement. No improvement or novelty is claimed yet.
