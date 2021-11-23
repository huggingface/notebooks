import argparse
import logging
import os
import sys

import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding, create_optimizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--learning_rate", type=str, default=3e-5)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load DatasetDict
    dataset = load_dataset("imdb")

    # Preprocess train dataset
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # define tokenizer_columns
    # tokenizer_columns is the list of keys from the dataset that get passed to the TensorFlow model
    tokenizer_columns = ["attention_mask", "input_ids"]

    # convert to TF datasets
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    encoded_dataset["train"] = encoded_dataset["train"].rename_column("label", "labels")
    tf_train_dataset = encoded_dataset["train"].to_tf_dataset(
        columns=tokenizer_columns,
        label_cols=["labels"],
        shuffle=True,
        batch_size=8,
        collate_fn=data_collator,
    )
    encoded_dataset["test"] = encoded_dataset["test"].rename_column("label", "labels")
    tf_validation_dataset = encoded_dataset["test"].to_tf_dataset(
        columns=tokenizer_columns,
        label_cols=["labels"],
        shuffle=False,
        batch_size=8,
        collate_fn=data_collator,
    )

    # Prepare model labels - useful in inference API
    labels = encoded_dataset["train"].features["labels"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # download model from model hub
    model = TFAutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    # create Adam optimizer with learning rate scheduling
    batches_per_epoch = len(encoded_dataset["train"]) // args.train_batch_size
    total_train_steps = int(batches_per_epoch * args.epochs)

    optimizer, _ = create_optimizer(init_lr=args.learning_rate, num_warmup_steps=0, num_train_steps=total_train_steps)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # define metric and compile model
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Training
    logger.info("*** Train ***")
    train_results = model.fit(
        tf_train_dataset,
        epochs=args.epochs,
        validation_data=tf_validation_dataset,
    )

    output_eval_file = os.path.join(args.output_data_dir, "train_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** Train results *****")
        logger.info(train_results)
        for key, value in train_results.history.items():
            logger.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))

    # Save result
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
