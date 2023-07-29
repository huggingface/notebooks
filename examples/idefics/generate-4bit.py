import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b"
#checkpoint = "HuggingFaceM4/tiny-random-idefics"

model = IdeficsForVisionText2Text.from_pretrained(checkpoint, load_in_4bit=True)
processor = AutoProcessor.from_pretrained(checkpoint)

prompts = [
    "Instruction: provide an answer to the question. Use the image to answer.\n",
    "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg",
    "Question: What's on the picture? Answer: \n"
]

inputs = processor(prompts, return_tensors="pt")
generated_ids = model.generate(**inputs, max_length=150)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text[0])
