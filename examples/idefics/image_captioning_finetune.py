# adapted from https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/image_captioning.ipynb

# This example demonstrates normal finetuning (w/o peft) - for the sake of keeping the memory
# requirements small it freezes the original pre-trained text and image layers to keep the memory
# requirements to just 40GB. If you have multiple GPUs then you can remove the unfreeze part to
# finetune the whole model. Alternatively use the PEFT solution as shown in
# IDEFICS_finetuning_demo.ipynb notebook which requires only 20GB to finetune the whole model.

import torch
import torchvision.transforms as transforms

from datasets import load_dataset
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b"
# checkpoint = "HuggingFaceM4/tiny-random-idefics"

processor = AutoProcessor.from_pretrained(checkpoint)
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)

# freeze the original text and vision models and finetune only the layers added by IDEFICS
# you can unfreeze the whole model, but it'll require multiple gpus to finetune
model.model.freeze_text_layers()
model.model.freeze_vision_layers()

# help util
def check_inference():
    url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
    prompts = [
        url,
        "Question: What's on the picture? Answer:",
    ]

    inputs = processor(prompts, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_length=150)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)

# check generation before finetuning
check_inference()
# well, actually it looks like the model is already aware of pokemon - but this dataset will refine it further

# finetune the model on the pokemon types dataset
ds = load_dataset("GabeHD/pokemon-type-captions")
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
eval_ds = ds["test"]

def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

def ds_transforms(example_batch):
    image_size = processor.image_processor.image_size
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    image_transform = transforms.Compose([
        convert_to_rgb,
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    prompts = []
    for i in range(len(example_batch)):
        prompts.append(
            [
                example_batch["image"][i],
                f"Question: What's on the picture? Answer: {example_batch['text'][i]}\n",
            ],
        )

    inputs = processor(prompts, transform=image_transform, return_tensors="pt").to(device)

    inputs["labels"] = inputs["input_ids"]

    return inputs

train_ds.set_transform(ds_transforms)
eval_ds.set_transform(ds_transforms)

model_name = checkpoint.split("/")[1]

# this setup requires about 40GB of gpu memory
training_args = TrainingArguments(
    output_dir=f"{model_name}-pokemon",
    learning_rate=5e-6,
    num_train_epochs=10,
    bf16=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    dataloader_pin_memory=False,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=1000, # don't save until ready...
    eval_steps=40,
    logging_steps=40,
    remove_unused_columns=False,
    push_to_hub=False,
    label_names=["labels"],
    load_best_model_at_end=True,
    report_to=None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train()

# check generation again after finetuning
check_inference()

# after finetuning ideally we want generate to produce something like: a drawing of a pink and blue pokemon
