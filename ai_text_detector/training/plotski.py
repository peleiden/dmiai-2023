import json
import os
from glob import glob
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pelutils import Flag, JobDescription, Option, Parser, Union, log
from pelutils.ds.plots import exp_moving_avg, double_moving_avg

plt.rcParams["figure.figsize"] = (10, 5)


def setup_plot(name, path, args: JobDescription):
    plt.xlabel("Epoch")
    if "loss" in name:
        plt.ylabel("Cross-entropy loss")
        plt.yscale("log")
    if "rouge" in name:
        plt.ylim(bottom=0)
        plt.ylabel("ROUGE [%]")

    plt.title(name.replace("_", " ").capitalize())
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{name}.{'pdf' if args.pdf else 'png'}"))
    plt.clf()


def plot_single(
    x: Tuple[float, ...],
    y: Tuple[float, ...],
    args: JobDescription,
    label: Optional[str] = None,
):
    if len(x) < 25:
        plt.scatter(x, y)
        plt.plot(x, y, label=label, ls="--")
    else:
        plt.plot(x, y, label=label)
    if args.max_x_axis > 0:
        plt.xticks(np.arange(0, args.max_x_axis + 1))


def single_run(
    run_path: str,
    args: JobDescription,
) -> Union[None, Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]]]:
    # Sort after # steps and take newest checkpoint
    try:
        chk_path = sorted(
            [path for path in glob(f"{run_path}/checkpoint-*") if not path.endswith("best")],
            key=lambda s: int(s.split("-")[-1]),
        )[-1]
    except IndexError:
        log.warning(f"Could not find checkpoint in {run_path}.")
        return None

    log(f"Plotting for {chk_path} ...")
    with open(
        os.path.join(chk_path, "trainer_state.json"),
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)
    log(
        "Training metadata",
        json.dumps(
            {
                key: data.get(key)
                for key in ("max_steps", "num_train_epochs", "total_flos")
            },
            indent=2,
        ),
    )

    plot_path = os.path.join(run_path, "plots")
    os.makedirs(plot_path, exist_ok=True)
    # Include all metrics that appear in at least one iteration
    out_data = {}
    for metric in sorted(
        {
            metric
            for iteration in data["log_history"]
            for metric, value in iteration.items()
            if value and not np.isnan(value)
        }
    ):
        if metric not in {"epoch", "step"}:
            x_axis = "step" if args.steps else "epoch"
            x, y = zip(
                *[
                    (iteration.get(x_axis), iteration.get(metric))
                    for iteration in data["log_history"]
                    if iteration.get(metric)
                ]
            )
            x, y = zip(
                *[
                    (x_, y_)
                    for x_, y_ in zip(x, y)
                    if (args.max_x_axis < 0 or x_ <= args.max_x_axis)
                ]
            )
            if metric == "loss":
                x, y = double_moving_avg(np.array(x), np.array(y))
            plot_single(x, y, args)
            setup_plot(metric, plot_path, args)
            out_data[metric] = x, y
    return out_data


def run(args: JobDescription):
    log("Reporting on training job with args", args)
    run_paths = sorted(glob(f"{args.location}/*/*-ai-detector"))
    # Do individual plots
    run_results = [single_run(run_path, args) for run_path in run_paths]
    run_results = [result for result in run_results if result is not None]

    # Do plot of all examples
    plot_path = os.path.join(args.location, "plots")
    os.makedirs(plot_path, exist_ok=True)
    for metric in run_results[0]:
        for path, result in zip(run_paths, run_results):
            if (res := result.get(metric)) is not None:
                plot_single(
                    res[0],
                    res[1],
                    args,
                    label=os.path.split(os.path.dirname(path))[-1],
                )
            else:
                log.warning(
                    f"Did not find metric {metric} for experiment {path}"
                )
        plt.legend()
        setup_plot(metric, plot_path, args)


if __name__ == "__main__":
    job: JobDescription = Parser(
        Option("max-x-axis", default=-1),
        Flag("pdf"),
        Flag("steps"),
    ).parse_args()
    log.configure(
        os.path.join(job.location, "train-report.log"),
    )
    log.log_repo()
    with log.log_errors:
        run(job)
