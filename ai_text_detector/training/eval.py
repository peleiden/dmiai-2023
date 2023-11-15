from collections import defaultdict
from glob import glob as glob # glob
from dataclasses import dataclass
import os
import torch
from pelutils import DataStorage, JobDescription, Option, Parser, log, Table
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification, PreTrainedModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
import numpy as np

from ai_text_detector.training.hf_loop import get_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Classifier:
    def __init__(self, model_paths: list[str]):
        checkpoints = [self._get_best_chk(model_path) for model_path in model_paths]

        self.models: list[PreTrainedModel] = [
            # FIXME: Change to chk
            BertForSequenceClassification.from_pretrained("hf-internal-testing/tiny-bert").eval().to(DEVICE) for chk in checkpoints
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(self.models[0].config._name_or_path)

    @staticmethod
    def _get_best_chk(path: str):
        return sorted(
            glob(f"{path}/checkpoint-*"),
            key=lambda s: float("inf") if "best" in s else int(s.split("-")[-1]),
        )[-1]


    def predict(self, texts: list[str], batch_size: int) -> list[int]:
        return [int(prob > .5) for prob in self.predict_prob(texts, batch_size)]

    def predict_probs(self, texts: list[str], batch_size: int, vote=True):
        for batch in tqdm(DataLoader(texts, batch_size=batch_size)):
            inputs = self.tokenizer(batch, max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                outputs = torch.stack([model(**inputs).logits for model in self.models])
            probs = torch.nn.functional.softmax(outputs, -1)[..., -1]
            for i in range(probs.shape[1]):
                if vote:
                    yield float(probs[:, i].mean())
                else:
                    yield [float(p) for p in probs[:, i]]

@dataclass
class EvalResults(DataStorage):
    texts: list[str]
    # contains probabilities that texts are positive class in
    # (number of texts x number of models in ensemble)
    model_probs: np.ndarray
    overall_scores: dict[str, float]
    model_scores: list[dict[str, float]]

def wilson_score_interval(p, n, z=1.96):  # z for 95% CI
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2*n)
    adjusted_standard_deviation = np.sqrt((p*(1 - p) + z**2 / (4*n)) / n)

    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return lower_bound, upper_bound


def compute_all_scores(probs: np.ndarray, true) -> dict[str, float]:
    preds = probs > .5
    true = np.array(true)
    scores =  dict(accuracy = accuracy_score(true, preds),
        F1 = f1_score(true, preds),
        true_pos_rate = recall_score(true, preds),
        true_neg_rate = ((preds == true) & (preds == 0)).sum() / (true == 1).sum(),
        precision = precision_score(true, preds),
        roc_auc = roc_auc_score(true, probs),
        TP = ((preds == true) & (preds == 1)).sum(),
        TN = ((preds == true) & (preds == 0)).sum(),
        FP = ((preds != true) & (preds == 1)).sum(),
        FN = ((preds != true) & (preds == 0)).sum(),
    )
    return {name: float(score) for name, score in scores.items()}

def format_scores(scores: dict[str, float], n: int):
    t = Table()
    t.add_header(["Metric", "Score", "95% CI [%]"])
    for name, score in scores.items():
        uncertainty = ("%.1f; %.1f" % tuple(100*x for x in wilson_score_interval(score, n))) if score <= 1 else ""
        t.add_row([name, ("%.1f" % (score * 100)) if score < 1 else int(score),  uncertainty])
    return t

def aggregate_fold_scores(score_dicts: list[dict[str, float]]):
    t = Table()
    t.add_header(["Metric", "Score", "95% CI [%]"])
    for metric in score_dicts[0]:
        scores = [sd[metric] for sd in score_dicts]
        score = np.mean(scores)
        uncertainty = 1.96 * (np.std(scores, ddof=1) / np.sqrt(len(scores)))
        t.add_row([metric, ("%.1f" % (score * 100)) if score < 1 else int(score),  uncertainty])
    return t


def eval(args: JobDescription):
    base_name = args.base_model.split('/')[-1]
    all_results = []
    for i, path in enumerate(sorted(glob(f"{args.location}/fold*"))):
        dataset = get_data(args, i, do_tokenize=False)
        log.section("Evaluating model %i" % i)
        if (model_paths := glob(f"{args.location}/{base_name}-*-ai-detector")):
            model = Classifier(model_paths)
        else:
            model = Classifier([os.path.join(path, f"{base_name}-ai-detector")])
        log(f"Loaded {len(model.models)} models")
        model_probs = np.array(list(
            model.predict_probs(dataset["test"]["text"], batch_size=args.batch_size, vote=False)
        ))
        results = EvalResults(dataset["test"]["text"], model_probs=model_probs, overall_scores={}, model_scores=[{}])
        results.save(save_path := os.path.join(path, "eval-results"))
        all_results.append(results)

        results.model_scores = [compute_all_scores(probs, dataset["test"]["label"]) for probs in model_probs.T]
        results.overall_scores = compute_all_scores(model_probs.mean(axis=1), dataset["test"]["label"])
        results.save(save_path)


        if len(model.models) > 1:
            for i, model in enumerate(model.models):
                log("Scores for single ensemble model %i" % i, format_scores(results.model_scores[i], len(dataset)))
            for n_models in range(2, len(model.models) + 1):
                # TODO: Average across different models
                ensemble_scores = compute_all_scores(model_probs[:, :n_models].mean(axis=1), dataset["test"]["label"])
                log("Scores for ensemble with %i models" % n_models, format_scores(ensemble_scores, len(dataset)))

        log("Overall model scores", format_scores(results.overall_scores, len(dataset)))

    log("Scores across all folds", aggregate_fold_scores([res.overall_scores for res in all_results]))



if __name__ == "__main__":
    parser = Parser(
        Option(
            "base-model",
            default="chcaa/dfm-encoder-large-v1",
            help="The huggingface model from which to initialize parameters",
        ),
        Option("batch-size", type=int, default=32),
        Option("seed", default=0),
        Option("cv-folds", default=10),
    )

    job: JobDescription = parser.parse_args()
    log.configure(
        os.path.join(job.location, "ai-detector-eval.log"),
        append=True,
    )
    log.log_repo()
    log(f"Starting {job.name}")
    with log.log_errors:
        eval(job)
