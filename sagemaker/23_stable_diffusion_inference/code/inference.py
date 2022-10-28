import base64
import torch
from io import BytesIO
from diffusers import StableDiffusionPipeline


def model_fn(model_dir):
  # Load stable diffusion and move it to the GPU
  pipe = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
  pipe = pipe.to("cuda")

  return pipe

def predict_fn(data, pipe):

    # get prompt & parameters
    prompt = data.pop("inputs", data)
    # set valid HP for stable diffusion
    num_inference_steps = data.pop("num_inference_steps", 50)
    guidance_scale = data.pop("guidance_scale", 7.5)
    num_images_per_prompt = data.pop("num_images_per_prompt", 4)

    # run generation with parameters
    generated_images = pipe(prompt,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,num_images_per_prompt=num_images_per_prompt)["images"]
    
    # create response 
    encoded_images=[]
    for image in generated_images:
      buffered = BytesIO()
      image.save(buffered, format="JPEG")
      encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    # create response
    return {"generated_images": encoded_images}
