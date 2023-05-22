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
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DonutProcessor, VisionEncoderDecoderModel,VisionEncoderDecoderConfig
import shutil
import logging
import sys 
import json 

def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        default="naver-clova-ix/donut-base",
        help="Model id to use for training.",
    )
    parser.add_argument("--special_tokens", type=str, default=None, help="JSON string of special tokens to add to tokenizer.")
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
        default=False,
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


def training_function(args):
    # set seed
    set_seed(args.seed)

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.dataset_path)
    image_size = list(torch.tensor(train_dataset[0]["pixel_values"][0]).shape) # height, width
    logger.info(f"loaded train_dataset length is: {len(train_dataset)}")

    # Load processor and set up new special tokens
    processor = DonutProcessor.from_pretrained(args.model_id)
    # add new special tokens to tokenizer and resize feature extractor
    special_tokens = args.special_tokens.split(",")
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    processor.feature_extractor.size = image_size[::-1] # should be (width, height)
    processor.feature_extractor.do_align_long_axis = False

    # Load model from huggingface.co
    config = VisionEncoderDecoderConfig.from_pretrained(args.model_id, use_cache=False if args.gradient_checkpointing else True)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_id, config=config)
    
    # Resize embedding layer to match vocabulary size & adjust our image size and output sequence lengths
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.encoder.image_size = image_size
    model.config.decoder.max_length = len(max(train_dataset["labels"], key=len))
    # Add task token for decoder to start
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]


    # Arguments for training
    output_dir = "/tmp"
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=True,
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=10,
        save_total_limit=1,
        evaluation_strategy="no",
        save_strategy="epoch",
    )

    # Create Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()

    # save model and processor
    trainer.model.save_pretrained("/opt/ml/model/")
    processor.save_pretrained("/opt/ml/model/")
    
    # copy inference script
    os.makedirs("/opt/ml/model/code", exist_ok=True)
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "inference.py"),
        "/opt/ml/model/code/inference.py",
    )


def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
