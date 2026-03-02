#%%
from typing import List

import torch
from PIL import Image

from diffusers import AutoencoderKLWan, LucyEditPipeline
from diffusers.utils import export_to_video, load_video

#%%
# Arguments
url = "assets/lunette_test.mp4"
prompt = "Darken sunglasses so that reflections are no longer visible, but so that the eyes can still be seen"
negative_prompt = ""
num_frames = 52
height = 1080
width = 1920

# Load video
def convert_video(video: List[Image.Image]) -> List[Image.Image]:
    video = load_video(url)[:num_frames]
    video = [video[i].resize((width, height)) for i in range(num_frames)]
    return video

video = load_video(url, convert_method=convert_video)
#%%
# Load model
model_id = "decart-ai/Lucy-Edit-Dev"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = LucyEditPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
# pipe.to("cuda")

# Generate video
output = pipe(
    prompt=prompt,
    video=video,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0
).frames[0]

# Export video
export_to_video(output, "output.mp4", fps=24)

# %%
