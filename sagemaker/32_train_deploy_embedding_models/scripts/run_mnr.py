from dataclasses import dataclass, field
import os
from sentence_transformers import (
    SentenceTransformerModelCardData,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from transformers import set_seed, HfArgumentParser


from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim
from datasets import load_dataset, concatenate_datasets


@dataclass
class ScriptArguments:
    train_dataset_path: str = field(
        default="/opt/ml/input/data/train/",
        metadata={"help": "Path to the dataset, e.g. /opt/ml/input/data/train/"},
    )
    test_dataset_path: str = field(
        default="/opt/ml/input/data/test/",
        metadata={"help": "Path to the dataset, e.g. /opt/ml/input/data/test/"},
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for Embedding training"}
    )
    num_train_epochs: int = field(
        default=1, metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=32, metadata={"help": "Training batch size"}
    )
    per_device_eval_batch_size: int = field(
        default=16, metadata={"help": "Evaluation batch size"}
    )
    gradient_accumulation_steps: int = field(
        default=16, metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-5, metadata={"help": "Learning rate for the optimizer"}
    )


def create_evaluator(
    train_dataset, test_dataset, matryoshka_dimensions=[768, 512, 256, 128, 64]
):
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

    # Convert the datasets to dictionaries
    corpus = dict(
        zip(corpus_dataset["id"], corpus_dataset["positive"])
    )  # Our corpus (cid => document)
    queries = dict(
        zip(test_dataset["id"], test_dataset["anchor"])
    )  # Our queries (qid => question)

    # Create a mapping of relevant document (1 in our case) for each query
    relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
    for q_id in queries:
        relevant_docs[q_id] = [q_id]

    matryoshka_evaluators = []
    # Iterate over the different dimensions
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Truncate the embeddings to a certain dimension
            score_functions={"cosine": cos_sim},
        )
        matryoshka_evaluators.append(ir_evaluator)

    # Create a sequential evaluator
    return SequentialEvaluator(matryoshka_evaluators)


def training_function(script_args):
    ################
    # Dataset
    ################

    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.train_dataset_path, "dataset.json"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.test_dataset_path, "dataset.json"),
        split="train",
    )

    ###################
    # Model & Evaluator
    ###################

    matryoshka_dimensions = [768, 512, 256, 128, 64]  # Important: large to small

    model = SentenceTransformer(
        script_args.model_id,
        device="cuda",
        model_kwargs={"attn_implementation": "sdpa"},  # needs Ampere GPU or newer
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="BGE base Financial Matryoshka",
        ),
    )
    evaluator = create_evaluator(
        train_dataset, test_dataset, matryoshka_dimensions=matryoshka_dimensions
    )

    ###################
    # Loss Function
    ###################

    # create Matryoshka loss function with MultipleNegativesRankingLoss
    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
    )

    ################
    # Training
    ################
    training_args = SentenceTransformerTrainingArguments(
        output_dir="/opt/ml/model",  # output directory for sagemaker to upload to s3
        num_train_epochs=script_args.num_train_epochs,  # number of epochs
        per_device_train_batch_size=script_args.per_device_train_batch_size,  # training batch size
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,  # evaluation batch size
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,  # gradient accumulation steps
        warmup_ratio=0.1,  # warmup ratio
        learning_rate=script_args.learning_rate,  # learning rate
        lr_scheduler_type="cosine",  # use constant learning rate scheduler
        optim="adamw_torch_fused",  # use fused adamw optimizer
        tf32=True,  # use tf32 precision
        bf16=True,  # use bf16 precision
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        eval_strategy="epoch",  # evaluate after each epoch
        save_strategy="epoch",  # save after each epoch
        logging_steps=10,  # log every 10 steps
        save_total_limit=3,  # save only the last 3 models
        load_best_model_at_end=True,  # load the best model when training ends
        metric_for_best_model="eval_dim_128_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score for the 128 dimension
    )

    trainer = SentenceTransformerTrainer(
        model=model,  # bg-base-en-v1
        args=training_args,  # training arguments
        train_dataset=train_dataset.select_columns(
            ["positive", "anchor"]
        ),  # training dataset
        loss=train_loss,
        evaluator=evaluator,
    )

    ##########################
    # Train model
    ##########################
    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()

    # save the best model
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments))
    script_args = parser.parse_args_into_dataclasses()[0]

    # set seed
    set_seed(42)

    # launch training
    training_function(script_args)
