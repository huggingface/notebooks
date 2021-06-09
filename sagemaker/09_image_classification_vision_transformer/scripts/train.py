from transformers import ViTForImageClassification, Trainer, TrainingArguments,default_data_collator,ViTFeatureExtractor
from datasets import load_from_disk,load_metric
import random
import logging
import sys
import argparse
import os
import numpy as np
import subprocess

subprocess.run([
        "git",
        "config",
        "--global",
        "user.email",
        "sagemaker@huggingface.co",
    ], check=True)
subprocess.run([
        "git",
        "config",
        "--global",
        "user.name",
        "sagemaker",
    ], check=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_dir", type=str,default="/opt/ml/model")
    parser.add_argument("--extra_model_name", type=str,default="sagemaker")
    parser.add_argument("--dataset", type=str,default="cifar10")
    parser.add_argument("--task", type=str,default="image-classification")
    parser.add_argument("--use_auth_token", type=str, default="")

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=str, default=2e-5)

    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)
    num_classes = train_dataset.features["label"].num_classes


    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    metric_name = "accuracy"
    # compute metrics function for binary classification

    metric = load_metric(metric_name)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # download model from model hub
    model = ViTForImageClassification.from_pretrained(args.model_name,num_labels=num_classes)
    
    # change labels
    id2label =  {key:train_dataset.features["label"].names[index] for index,key in enumerate(model.config.id2label.keys())}
    label2id =  {train_dataset.features["label"].names[index]:value for index,value in enumerate(model.config.label2id.values())}
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    
    # define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )
    
    
    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=default_data_collator,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.output_dir)

    if args.use_auth_token != "":
        kwargs = {
            "finetuned_from": args.model_name.split("/")[1],
            "tags": "image-classification",
            "dataset": args.dataset,
        }
        repo_name = (
            f"{args.model_name.split('/')[1]}-{args.task}"
            if args.extra_model_name == ""
            else f"{args.model_name.split('/')[1]}-{args.task}-{args.extra_model_name}"
        )
 
        trainer.push_to_hub(
            repo_name=repo_name,
            use_auth_token=args.use_auth_token,
            **kwargs,
        )
