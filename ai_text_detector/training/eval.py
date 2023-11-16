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
    threshold = .5

    def __init__(self, model_paths: list[str], tokenizer_key: str="chcaa/dfm-encoder-large-v1"):
        checkpoints = [self._get_best_chk(model_path) for model_path in model_paths]
        assert checkpoints

        self.models: list[PreTrainedModel] = [
            BertForSequenceClassification.from_pretrained(chk).eval().to(DEVICE) for chk in checkpoints if chk
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_key)

    @staticmethod
    def _get_best_chk(path: str):
        paths = sorted(
            glob(f"{path}/checkpoint-*"),
            key=lambda s: float("inf") if "best" in s else int(s.split("-")[-1]),
        )
        return paths[-1] if paths else None


    def predict(self, texts: list[str], batch_size: int) -> list[int]:
        return [int(prob > self.threshold) for prob in self.predict_probs(texts, batch_size)]

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
    true_labels: list[int]
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


def compute_all_scores(probs: np.ndarray, true, threshold: float) -> dict[str, float]:
    preds = (probs > threshold).astype(int)
    true = np.array(true)
    scores =  dict(accuracy = accuracy_score(true, preds),
        F1_score = f1_score(true, preds),
        true_pos_rate = recall_score(true, preds),
        true_neg_rate = ((preds == true) & (preds == 0)).sum() / (true == 0).sum(),
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
        uncertainty = ("%.1f; %.1f" % tuple(100*x for x in wilson_score_interval(score, n))) if not name.isupper() else ""
        t.add_row([name, ("%.2f" % (score * 100)) if not name.isupper() else int(score),  uncertainty])
    return t

def aggregate_fold_scores(score_dicts: list[dict[str, float]]):
    t = Table()
    t.add_header(["Metric", "Score", "95% CI [%]"])
    for metric in score_dicts[0]:
        scores = [sd[metric] for sd in score_dicts]
        score = np.mean(scores)
        moe = 1.96 * (np.std(scores, ddof=1) / np.sqrt(len(scores)))
        conf = (score - moe, score + moe)
        uncertainty = "%.1f; %.1f" % tuple((1 if metric.isupper() else 100) * x for x in conf)
        t.add_row([metric, "%.2f" % (score * (1 if metric.isupper() else 100)),  uncertainty])
    return t


def eval(args: JobDescription):
    base_name = args.base_model.split('/')[-1]
    all_results = []
    for i, path in enumerate(sorted(glob(os.path.join(args.location, "fold*")))):
        save_path = os.path.join(path, "eval-results")
        log.section("Evaluating model %i" % i)
        try:
            results = EvalResults.load(save_path)
        except FileNotFoundError:
            dataset = get_data(args, i, do_tokenize=False)
            if (model_paths := glob(os.path.join(path, f"{base_name}-idx*-ai-detector"))):
                model = Classifier(sorted(model_paths))
            else:
                model = Classifier([os.path.join(path, f"{base_name}-ai-detector")])
            model.threshold = args.threshold
            log(f"Loaded {len(model.models)} models")
            model_probs = np.array(list(
                model.predict_probs(dataset["test"]["text"], batch_size=args.batch_size, vote=False)
            ))
            results = EvalResults(dataset["test"]["text"], model_probs=model_probs, overall_scores={}, model_scores=[{}], true_labels=dataset["test"]["label"])
            results.save(save_path)
        results.model_scores = [compute_all_scores(probs, results.true_labels, args.threshold) for probs in results.model_probs.T]
        results.overall_scores = compute_all_scores(results.model_probs.mean(axis=1), results.true_labels, args.threshold)
        results.save(save_path)
        all_results.append(results)


        total_models = results.model_probs.shape[1]
        if total_models > 1:
            for i in range(total_models):
                log("Scores for single ensemble model %i" % i, format_scores(results.model_scores[i], len(results.texts)))
            for n_models in range(2, total_models):
                # TODO: Average across different models pairs
                ensemble_scores = compute_all_scores(results.model_probs[:, :n_models].mean(axis=1), results.true_labels, args.threshold)
                log("Scores for ensemble with %i models" % n_models, format_scores(ensemble_scores, len(results.texts)))

        log("Overall model scores for %i models" % total_models, format_scores(results.overall_scores, len(results.texts)))

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
        Option("threshold", default=.5),
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
