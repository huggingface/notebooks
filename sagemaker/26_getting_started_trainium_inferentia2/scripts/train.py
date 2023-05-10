import os
import argparse
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from datasets import load_from_disk
import evaluate
import numpy as np
import logging 
from transformers import TrainingArguments
from optimum.neuron import TrainiumTrainer as Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

login(token="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")

logger = logging.getLogger(__name__)


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default="bert-large-uncased", help="Model id to use for training.")
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Hugging Face Repository id for uploading models"
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument(
        "--bf16",
        type=bool,
        default=False,
        help="Whether to use bf16.",
    )
    args = parser.parse_known_args()
    return args

# Metric Id
metric = evaluate.load("f1")

# Metric helper method
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")


def training_function(args):
    # set seed
    set_seed(args.seed)

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f"loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f"loaded test_dataset length is: {len(test_dataset)}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Prepare model labels - useful for inference
    labels = train_dataset.features["labels"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Download the model from huggingface.co/models
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    # Define training args
    output_dir = args.model_id.split("/")[-1] if "/" in args.model_id else args.model_id
    output_dir = f"{output_dir}-finetuned"
    training_args = TrainingArguments(
        overwrite_output_dir=True,
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="tensorboard",
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # convert model to inferentia2
    with training_args.main_process_first(desc="saving model and tokenizer"):
        # tokenizer  
        trainer.model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)



def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()