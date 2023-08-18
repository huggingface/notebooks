# this is a demo of inference of IDEFICS-9B which needs about 20GB of GPU memory

import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b"
#checkpoint = "HuggingFaceM4/tiny-random-idefics"

model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)

url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
image = processor.image_processor.fetch_images(url)

prompts = [
    [
        "User:",
        image,
        "Describe this image.\nAssistant: An image of two kittens in grass.\n",
        "User:",
        "https://hips.hearstapps.com/hmg-prod/images/dog-puns-1581708208.jpg",
        "Describe this image.\nAssistant:",
    ],
    [
        "User:",
        "https://hips.hearstapps.com/hmg-prod/images/dog-puns-1581708208.jpg",
        "Describe this image.\nAssistant: An image of a dog wearing funny glasses.\n",
        "User:",
        "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg",
        "Describe this image.\nAssistant:",
    ],
    [
        "User:",
        image,
        "Describe this image.\nAssistant: An image of two kittens in grass.\n",
        "User:",
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image1.jpeg",
        "Describe this image.\nAssistant:",
    ],
    [
        "User:",
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image2.jpeg",
        "Describe this image.\nAssistant: An image of a dog.\n",
        "User:",
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image1.jpeg",
        "Describe this image.\nAssistant:",
    ],
]

# batched mode
inputs = processor(prompts, return_tensors="pt").to(device)
# single sample mode
#inputs = processor(prompts[0], return_tensors="pt").to(device)

generated_ids = model.generate(**inputs, max_length=128)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
for i,t in enumerate(generated_text):
    print(f"{i}:\n{t}\n")
