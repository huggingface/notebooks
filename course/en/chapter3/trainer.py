import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForSequenceClassification

### dataset, checkpoint  
raw_datasets = load_dataset("glue", "sst2")
checkpoint = "bert-base-uncased"

### tokenize
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

### metrics
def compute_metrics(eval_preds):  # takes on EvalPrediction object
    metric = evaluate.load("glue", "sst2")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

### model
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch", push_to_hub=False)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # <-- we need this to evaluate!!!
)

trainer.train()
