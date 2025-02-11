from diffusers import LTXPipeline
model_id = "Lightricks/LTX-Video"
cache_dir = './pretrained_model'
pipe = LTXPipeline.from_pretrained(model_id, cache_dir=cache_dir)