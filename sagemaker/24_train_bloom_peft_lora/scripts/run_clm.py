import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
)
from datasets import load_from_disk
import torch
from transformers import Trainer, TrainingArguments
from peft import PeftConfig, PeftModel
import shutil


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/flan-t5-xl",
        help="Model id to use for training.",
    )
    parser.add_argument("--dataset_path", type=str, default="lm_dataset", help="Path to dataset.")
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    args = parser.parse_known_args()
    return args


def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query_key_value"],
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def training_function(args):
    # set seed
    set_seed(args.seed)

    dataset = load_from_disk(args.dataset_path)
    # load model from the hub
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
        device_map="auto",
        load_in_8bit=True,
    )
    # create peft config
    model = create_peft_config(model)

    # Define training args
    output_dir = "/tmp"
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=2,
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        optim="adafactor",
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    # Start training
    trainer.train()

    # merge adapter weights with base model and save
    # save int 8 model
    trainer.model.save_pretrained(output_dir)
    # clear memory
    del model
    del trainer
    # load PEFT model in fp16
    peft_config = PeftConfig.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, output_dir)
    model.eval()
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("/opt/ml/model/")

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained("/opt/ml/model/")

    # copy inference script
    os.makedirs("/opt/ml/model/code", exist_ok=True)
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "inference.py"),
        "/opt/ml/model/code/inference.py",
    )
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt"),
        "/opt/ml/model/code/requirements.txt",
    )


def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
