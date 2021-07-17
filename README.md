VCLIP is an extension of OpenAI's CLIP for variational inference. It was fine-tuned on a subset of [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/). This repo contains the simple implementation code and a link to the weights. This implementation is an extension of the HuggingFace [FlaxCLIPModel](https://huggingface.co/transformers/model_doc/clip.html#flaxclipmodel).

[Pretrained weights download (Google Cloud Storage)](https://storage.googleapis.com/noahtren-public/vclip.ckpt).

---

VCLIP computes a Gaussian distribution over images for each prompt, rather than returning a single point. The similarity function between `(text, img)` is the normal probability density function rather than cosine similarity.

![CLIP and VCLIP](/model-comparison.png)
left: CLIP, right: VCLIP
