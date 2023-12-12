from typing import Union
from PIL import Image
import base64
from fastapi import FastAPI
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers import AutoPipelineForText2Image
import torch
from io import BytesIO
import os
import gc
from datetime import datetime
from pydantic import BaseModel
from aliyun import MyAliyun


model_dir = "/workspace/models/"
access_token = os.environ.get("HG_ACCESS_TOKEN")


if model_dir:
    model_key_base = os.path.join(model_dir, "sdxl-turbo")
    model_key_refiner = os.path.join(model_dir, "sdxl-turbo")
else:
    model_key_base = "stabilityai/sdxl-turbo"
    model_key_refiner = "stabilityai/sdxl-turbo"



app = FastAPI()

enable_refiner = os.getenv("ENABLE_REFINER", "true").lower() == "true"
output_images_before_refiner = True

#pipe = StableDiffusionXLPipeline.from_pretrained(model_key_base, torch_dtype=torch.float16, use_auth_token=access_token, variant="fp16", use_safetensors=True)
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

pipe.to("cuda")

pipe.enable_xformers_memory_efficient_attention()


def save_to_oss(remote_path, local_img_path):
    oss = MyAliyun()
    oss.web_insert_aliyun_file(remote_path, local_img_path)
    print("Image saved to: ", remote_path)


class Item(BaseModel):
    remote_dir: str
    prompt: str
    num_images: int


app.get("/ping")
def ping():
    return {"data":"pong"}


@app.post("/generate")
def infer(item: Item):
    print("item: ", item, type(item), item.remote_dir)
    remote_dir, prompt, num_images = item.remote_dir, item.prompt, item.num_images
    scale = 9
    samples = 1
    steps = 1
    refiner_strength =0.3
    print("propmt: ", prompt)
    print("negative: ", negative)
    print("num_images: ", num_images)
    prompt, negative = [prompt] * samples, [negative] * samples
    images_url_list = []
    for i in range(0, num_images):
        images = pipe(prompt=prompt, guidance_scale=scale, num_inference_steps=steps).images
        os.makedirs(r"stable-diffusion-xl-turbo/outputs", exist_ok=True)
        gc.collect()
        torch.cuda.empty_cache()
        for i, image in enumerate(images):
            buffered = BytesIO()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{i}.png"
            print("start save image: ", datetime.now())
            local_path = f"/workspace/code/stable-diffusion-sdxl-turbo-demo/stable-diffusion-sdxl-turbo-demo/outputs/{filename}"
            print("end save image: ", datetime.now())
            image.save(local_path, format="PNG")
            new_image = Image.open(local_path)
            new_image = new_image.resize((512, 512))
            new_image.save(local_path)
            oss_file_path = f"{remote_dir}{filename}"
            save_to_oss(oss_file_path, local_path)
            image_url = f"https://pailaimi-static.oss-cn-chengdu.aliyuncs.com/{oss_file_path}"
            images_url_list.append(image_url)
            os.remove(local_path)
    return images_url_list
