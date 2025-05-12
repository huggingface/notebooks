"""
On one node, launch with `deepspeed --num_gpus N idefics_zero3_finetuning.py`
by replacing N with the number of your GPUs

For several nodes, using Slurm, a template script is provided at
`examples/idefics/idefics_zero3_finetuning/slurm_script_idefics_zero3_finetuning_multinode.slurm`

For more information, follow the tutorial on using DeepSpeed with Transformers at
https://huggingface.co/docs/transformers/main_classes/deepspeed
"""

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, IdeficsForVisionText2Text, Trainer, TrainingArguments


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b"

processor = AutoProcessor.from_pretrained(checkpoint, use_auth_token=True)


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
    image_transform = transforms.Compose(
        [
            convert_to_rgb,
            transforms.RandomResizedCrop(
                (image_size, image_size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ]
    )
    prompts = []
    for i in range(len(example_batch["caption"])):
        # We split the captions to avoid having very long examples, which would require more GPU ram during training
        caption = example_batch["caption"][i].split(".")[0]
        try:
            # There are a handful of images that are not hosted anymore. This is a small (dummy) hack to skip these
            processor.image_processor.fetch_images(example_batch["image_url"][i])
        except Exception:
            print(
                "Warning: at least one image couldn't be retrieved from the internet in an example. Skipping the"
                " batch."
            )
        prompts.append(
            [
                example_batch["image_url"][i],
                f"Question: What's on the picture? Answer: This is {example_batch['name'][i]}. {caption}</s>",
            ],
        )
    inputs = processor(prompts, transform=image_transform, return_tensors="pt").to(device)
    inputs["labels"] = inputs["input_ids"]
    return inputs


# load and prepare dataset
ds = load_dataset("TheFusion21/PokemonCards")
ds = ds["train"].train_test_split(test_size=0.002)
train_ds = ds["train"]
eval_ds = ds["test"]
train_ds.set_transform(ds_transforms)
eval_ds.set_transform(ds_transforms)


# Important, define the training_args before the model
ds_config = {
    "communication_data_type": "fp32",
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": False,
        "reduce_bucket_size": "auto",
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": False,
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 2e9,
        "stage3_max_reuse_distance": 2e9,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
    },
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "steps_per_print": 2000000,
}
training_args = TrainingArguments(
    output_dir="idefics-pokemon",
    learning_rate=2e-4,
    bf16=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    # gradient_checkpointing=True,  # Uncomment if OOM
    dataloader_pin_memory=False,
    save_total_limit=3,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=40,
    eval_steps=20,
    logging_steps=20,
    max_steps=40,
    remove_unused_columns=False,
    push_to_hub=False,
    label_names=["labels"],
    load_best_model_at_end=True,
    report_to="none",
    optim="adamw_torch",
    deepspeed=ds_config,
)

model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

result = trainer.train()
print(result)  # Prints one per process - mostly here for sanity check
