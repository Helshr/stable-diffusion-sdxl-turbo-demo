from typing import Union
from PIL import Image
import base64
from fastapi import FastAPI
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers import AutoPipelineForText2Image
import torch
from io import BytesIO
from queue import Queue
from threading import Thread
import os
import gc
from datetime import datetime
from pydantic import BaseModel
from aliyun import MyAliyun
import concurrent.futures
import asyncio


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

#pipe = StableDiffusionXLPipeline.from_pretrained(model_key_base, torch_dtype=torch.float16, use_auth_token=access_token, variant="fp16",use_safetensors=True)
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

pipe.to("cuda")

pipe.enable_xformers_memory_efficient_attention()


# 队列用于存储生成好的图片路径
image_queue = Queue()


def generate_and_save_image(prompt, steps, remote_dir):
    print("here...")
    image = pipe(prompt=prompt, num_inference_steps=steps, strength=1, guidance_scale=0.0).images[0]
    os.makedirs(r"stable-diffusion-sdxl-turbo/outputs", exist_ok=True)
    print("start gc: ", datetime.now())
    gc.collect()
    torch.cuda.empty_cache()
    print("end gc: ", datetime.now())
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}.jpg"
    print("start save image: ", datetime.now())
    local_path = f"/workspace/code/stable-diffusion-sdxl-turbo-demo/stable-diffusion-sdxl-turbo/outputs/{filename}"
    print("end save image: ", datetime.now())
    image.resize((512, 512))
    image.save(local_path, format="JPEG", quality=60)
    oss_file_path = f"{remote_dir}{filename}"
    image_queue.put((oss_file_path, local_path))
    image_url = f"https://pailaimi-static.oss-cn-chengdu.aliyuncs.com/{oss_file_path}"
    return image_url


def save_to_oss(remote_path, local_img_path):
    oss = MyAliyun()
    oss.web_insert_aliyun_file(remote_path, local_img_path)
    print("Image saved to: ", remote_path)


def upload_to_oss():
    while True:
        oss_file_path, local_path = image_queue.get()
        save_to_oss(oss_file_path, local_path)
        image_url = f"https://pailaimi-static.oss-cn-chengdu.aliyuncs.com/{oss_file_path}"
        #images_url_list.append(image_url)
        os.remove(local_path)
        image_queue.task_done()


class Item(BaseModel):
    remote_dir: str
    prompt: str
    steps: int


app.get("/ping")
def ping():
    return {"data":"pong"}


@app.post("/generate_and_upload")
async def generate_and_upload(item: Item):
    print("here: ", item)
    # 启动线程生成和保存图片
    remote_dir, prompt, steps = item.remote_dir, item.prompt, item.steps
    print("prompt: ", prompt)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(generate_and_save_image, prompt, steps, remote_dir)
        image_url = await asyncio.to_thread(future.result)
    # 返回成功的响应
    return [image_url]


# 设置线程数量，根据需要调整
num_threads = 4

# 创建并启动上传线程
upload_threads = []
for _ in range(num_threads):
    thread = Thread(target=upload_to_oss)
    thread.daemon = True
    thread.start()
    upload_threads.append(thread)
