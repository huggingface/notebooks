import base64
import torch
import os 
import json
from io import BytesIO
from diffusers import AutoPipelineForText2Image

# HF_ADAPTER_IDS="{\"prithivMLmods/Canopus-LoRA-Flux-FaceRealism\": \"Canopus-LoRA-Flux-FaceRealism.safetensors\"}" python inference.py

# ADAPTER_IDS needs to be a JSON object with the adapter id as key and the adapter weight name as value
# e.g. {"prithivMLmods/Canopus-LoRA-Flux-FaceRealism": "Canopus-LoRA-Flux-FaceRealism.safetensors"}
ADAPTERS = json.loads(os.getenv("HF_ADAPTER_IDS", "{}"))

MODEL_ID = os.getenv("HF_MODEL_ID", "black-forest-labs/FLUX.1-schnell")


def model_fn(model_dir):
    """Load the model from Hugging Face and apply the LoRA weights if provided."""
    pipeline = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="balanced")
    if len(ADAPTERS.keys()) > 0:
        for adapter_id in ADAPTERS.keys():
            print(f"Loading adapter: {adapter_id}")
            pipeline.load_lora_weights(adapter_id, weight_name=ADAPTERS[adapter_id], adapter_name=adapter_id)

    return pipeline


def predict_fn(data, pipe):
    """Run the model with the provided data and return the generated images."""
    # get prompt & parameters
    prompt = data.pop("inputs", data)
    
    # check if adapter id is provided
    adapter_id = data.pop("adapter_id", None)
    # if adapter id is provided, set the adapter
    if ADAPTERS.get(adapter_id, None) is not None:
        print(f"Using adapter: {adapter_id}")
        pipe.set_adapters(adapter_id)
    else:
        print(f"No valid adapter id provided, using base model")
        pipe.disable_lora()
    
    # set valid HP for stable diffusion
    num_inference_steps = data.pop("num_inference_steps", 4) # only need 4 for schnell version, dev version needs 20-30 or so               
    guidance_scale = data.pop("guidance_scale", 0)  # must be 0.0 for schnell version, dev version can be 3.5    
    num_images_per_prompt = data.pop("num_images_per_prompt", 4)

    # run generation with parameters
    generated_images = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    )["images"]

    # create response
    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    # create response
    return {"generated_images": encoded_images}


model = model_fn(".")
payload = {"inputs": "Headshot of handsome young man, wearing dark gray sweater with buttons and big shawl collar, brown hair and short beard, serious look on his face, black background, soft studio lighting, portrait photography --ar 85:128 --v 6.0 --style rawHeadshot of handsome young man, wearing dark gray sweater with buttons and big shawl collar, brown hair and short beard, serious look on his face, black background, soft studio lighting, portrait photography --ar 85:128 --v 6.0 --style rawHeadshot of handsome young man, wearing dark gray sweater with buttons and big shawl collar, brown hair and short beard, serious look on his face, black background, soft studio lighting, portrait photography --ar 85:128 --v 6.0 --style raw"} # "adapter_id": "prithivMLmods/Canopus-LoRA-Flux-FaceRealism"}
result = predict_fn(payload, model)

for i, image in enumerate(result["generated_images"]):
    # save image to file
    with open(f"image_{i}.jpg", "wb") as f:
        f.write(base64.b64decode(image))
