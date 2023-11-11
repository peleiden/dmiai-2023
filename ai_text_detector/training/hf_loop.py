import os

from datasets import DatasetDict, load_dataset
from pelutils import JobDescription, Option, Parser, log
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)


def preprocess_data(features: dict, tokenizer: BertTokenizer):
    model_inputs = tokenizer(
        features["text"],
        max_length=512,
        truncation=True,
        # padding="max_length",
    )
    model_inputs["labels"] = features["label"]
    return model_inputs


def get_data(args: JobDescription) -> DatasetDict:
    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    dataset = load_dataset("json", data_files=os.path.join(args.location, "dataset.jsonl"))["train"]
    dataset = dataset.train_test_split(test_size=args.val_split, seed=0)
    tokenized_dataset = dataset.map(
        lambda calls: preprocess_data(calls, tokenizer),
        batched=True,
    )
    return tokenized_dataset


def get_model(args: JobDescription) -> BertForSequenceClassification:
    return BertForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2,
    )


def get_training_args(args: JobDescription) -> TrainingArguments:
    model_name = f"{args.base_model.split('/')[-1]}-ai-detector"
    out_dir = os.path.join(args.location, model_name)
    return TrainingArguments(
        out_dir,
        per_device_train_batch_size=args.device_batch_size,
        per_device_eval_batch_size=args.eval_device_batch_size,


        evaluation_strategy="epochs",
        save_strategy="epochs",
        save_total_limit=2,
        logging_dir=os.path.join(out_dir, "logs"),

        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
    )


def train(args: JobDescription):
    log("Training with", args)

    train_args = get_training_args(args)
    log("Trainer arguments", train_args.to_json_string())

    data = get_data(args)
    log(
        f"Loaded data with {len(data['train'])} training calls"
        f" and {len(data['test'])} eval calls"
    )

    model = get_model(args)
    log(f"Loaded model with {model.num_parameters()/1e6:.1f} M parameters")

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
    )

    log("Starting training ...")
    trainer.train()

    log("Training finished")
    trainer.evaluate()


if __name__ == "__main__":
    parser = Parser(
        Option(
            "base-model",
            default="chcaa/dfm-encoder-large-v1",
            help="The huggingface model from which to initialize parameters",
        ),
        Option("epochs", type=int, default=3),
        Option("device-batch-size", type=int, default=16),
        Option("eval-device-batch-size", type=int, default=32),
        Option("val-split", type=float, default=0.5),
        Option("lr", default=2e-5)
        Option("weight-decay", default=0.01)
        multiple_jobs=True,
    )

    jobs = parser.parse_args()
    parser.document()
    for job in jobs:
        log.configure(
            os.path.join(job.location, "ai-detector-train.log"),
            append=True,
        )
        log.log_repo()
        log(f"Starting {job.name}")
        with log.log_errors:
            train(job)
