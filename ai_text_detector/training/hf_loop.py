from functools import partial
import os
import json

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from pelutils import JobDescription, Option, Parser, log, Flag
import numpy as np
import pandas as pd
import evaluate
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
        padding="max_length",
    )
    model_inputs["labels"] = features["label"]
    return model_inputs


def get_data(args: JobDescription, my_fold: int, do_tokenize=True) -> DatasetDict:
    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    df = pd.read_csv("full-val.csv")
    df = df.drop(columns=[col for col in df.columns if col != "text"])
    with open("true_labels.thingy", "r", encoding="utf-8") as file:
        df["label"] = pd.Series(
            [int(line.split()[-1]) for line in file.readlines() if line]
        )
    df = df.dropna()
    df["label"] = df["label"].astype(int)
    if args.cv_folds == 1:
        dataset = Dataset.from_pandas(df)
        if args.final:
            dataset = DatasetDict(train=dataset)
        else:
            dataset = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
    else:
        df = df.sample(frac=1, random_state=args.seed)
        fold_length = int(len(df) / args.cv_folds)
        val_start_index = my_fold * fold_length
        val_end_index = val_start_index + fold_length
        if my_fold == args.cv_folds - 1:
            val_end_index = len(df)
        dataset = DatasetDict(
            test=Dataset.from_pandas(df.iloc[val_start_index:val_end_index]),
            train=Dataset.from_pandas(
                pd.concat([df.iloc[:val_start_index], df.iloc[val_end_index:]])
            ),
        )
        # Be gone, ugly pandas stuff!
        if "__index_level_0__" in dataset:
            dataset = dataset.remove_columns("__index_level_0__")

    dataset["train"] = concatenate_datasets([dataset["train"], Dataset.from_pandas(pd.read_csv("self-generated-data.csv"))])

    if do_tokenize:
        tokenized_dataset = dataset.map(
            lambda texts: preprocess_data(texts, tokenizer),
            batched=True,
        )
        return tokenized_dataset
    return dataset


def get_model(args: JobDescription) -> BertForSequenceClassification:
    return BertForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
    )


def get_training_args(args: JobDescription, model_idx: int) -> TrainingArguments:
    model_name = f"{args.base_model.split('/')[-1]}-idx{model_idx}-ai-detector"
    out_dir = os.path.join(args.location, model_name)
    return TrainingArguments(
        out_dir,
        per_device_train_batch_size=args.device_batch_size,
        per_device_eval_batch_size=args.eval_device_batch_size,
        evaluation_strategy="no" if args.final else "epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=1,
        logging_dir=os.path.join(out_dir, "logs"),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_prop,
        lr_scheduler_type=args.scheduler,
    )


def compute_metrics(eval_pred, metric):
    prediction_arrays, labels = eval_pred
    predictions = np.argmax(prediction_arrays, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def get_metrics_function():
    metric = evaluate.load("accuracy")
    return partial(compute_metrics, metric=metric)


def train(args: JobDescription):
    log("Training with", args)


    data = get_data(args, args.my_fold)
    log(
        f"Loaded data with {len(data['train'])} training texts" +
        ("" if args.final else f" and {len(data['test'])} eval texts")
    )

    for i in range(args.n_ensemble):
        log.section("Training model %i" % i)

        train_args = get_training_args(args, i)
        log("Trainer arguments", train_args.to_json_string())

        model = get_model(args)
        log(f"Loaded model with {model.num_parameters()/1e6:.1f} M parameters")

        trainer = Trainer(
            model=model,
            compute_metrics=get_metrics_function(),
            args=train_args,
            train_dataset=data["train"],
            eval_dataset=None if args.final else data["test"] ,
        )

        log("Starting training ...")
        trainer.train()

        log("Training finished")
        if not args.final:
            eval_res = trainer.evaluate()
            log("Eval scores:", json.dumps(eval_res, indent=1))


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
        Option("val-split", type=float, default=0.25),
        Option("lr", default=2e-5),
        Option("weight-decay", default=0.01),
        Option("seed", default=0),
        Option("cv-folds", default=1),
        Option("my-fold", default=0),
        Option("warmup-prop", default=0.0),
        Option("scheduler", default="linear"),
        Option("n-ensemble", default=1),
        Flag("final"),
        multiple_jobs=True,
    )

    jobs = parser.parse_args()
    for job in jobs:
        log.configure(
            os.path.join(job.location, "ai-detector-train.log"),
            append=True,
        )
        log.log_repo()
        log(f"Starting {job.name}")
        with log.log_errors:
            train(job)
