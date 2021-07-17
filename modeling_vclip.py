import flax
import flax.linen as nn
import jax.numpy as np

from transformers import FlaxCLIPModel, PretrainedConfig, CLIPConfig


class VCLIP(nn.Module):
  def setup(self):
    model_config = PretrainedConfig.from_pretrained(
        "openai/clip-vit-base-patch32")
    model_config = CLIPConfig(text_config_dict=model_config.text_config,
                              vision_config_dict=model_config.vision_config)
    hf_model = FlaxCLIPModel(model_config)
    self.module = hf_model.module
    self.projection_dim = hf_model.module.config.projection_dim
    self.text_sig = nn.Dense(features=self.projection_dim)

  def embed_text(self, text):
    position_ids = np.broadcast_to(
        np.arange(np.atleast_2d(text['input_ids']).shape[-1]),
        text['input_ids'].shape)
    text_outputs = self.module.text_model(**text,
                                          position_ids=position_ids,
                                          deterministic=False)
    text_pooled = text_outputs[1]
    text_mu = self.module.text_projection(text_pooled)
    text_sig = self.text_sig(text_pooled)
    return {
        'mu': text_mu,
        'sig': nn.sigmoid(text_sig / np.sqrt(self.projection_dim)) * 2
    }

  def embed_img(self, img):
    vision_outputs = self.module.vision_model(pixel_values=img,
                                              deterministic=False)
    vision_pooled = vision_outputs[1]
    vision = self.module.visual_projection(vision_pooled)
    return vision

  @nn.compact
  def __call__(self, text, img):
    return {"text": self.embed_text(text), "vision": self.embed_img(img)}

if __name__ == "__main__":
  clip_model = VCLIP()
  clip_params = flax.serialization.from_bytes(flax.optim.Adam(),
                                            open("./8a185d.ckpt",
                                                 "rb").read())['target']
