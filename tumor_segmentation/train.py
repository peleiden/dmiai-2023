import os
from pelutils import JobDescription, Option, Parser, log, Flag
import matplotlib.pyplot as plt
import pelutils.ds.plots as plots
import numpy as np
import torch
import torch.cuda.amp as amp
from pelutils import log
from transformers import get_linear_schedule_with_warmup
from PIL import Image
from pprint import pformat

from tumor_segmentation import device, TrainConfig, TrainResults
from tumor_segmentation.data import dice, get_data_files, load_data, split_train_test, dataloader as dataloader_, vote, get_augmentation_pipeline
from tumor_segmentation.model import TumorBoi
from tumor_segmentation.mask_to_border import mask_to_border


def plot(location: str, results: TrainResults, config: TrainConfig):
    log.section("Plotting")
    with plots.Figure(os.path.join(location, "train.png"), figsize=(20, 10), fontsize=18):
        plt.subplot(121)
        for j in range(config.num_models):
            plt.plot(results.train_loss[j], "-o", c=plots.tab_colours[0], label="Train" if j == 0 else None)
            plt.plot(results.test_batches, results.test_loss[j], "-o", c=plots.tab_colours[1], label="Test" if j == 0 else None)
        plt.grid()
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(122)
        for j in range(config.num_models):
            plt.plot(results.train_dice[j], "-o", c=plots.tab_colours[0], label="Train" if j == 0 else None)
            plt.plot(results.test_batches, results.test_dice[j], "-o", c=plots.tab_colours[1], label="Test" if j == 0 else None)
        plt.plot(results.test_batches, results.ensemble_dice, "-o", c=plots.tab_colours[2], label="Test ensemble")
        plt.grid()
        plt.xlabel("Batch")
        plt.ylabel("Dice")
        plt.legend()

def plot_samples(path: str, ims: list[np.ndarray], segs: list[np.ndarray], pred_segs: list[np.ndarray], *, train: bool) -> list[np.ndarray]:
    ims = [im.copy() for im in ims]
    for i, (im, seg, pred) in enumerate(zip(ims, segs, pred_segs, strict=True)):
        true_border = mask_to_border(seg, padding=4 if train else 7)
        pred_border = mask_to_border(pred, padding=4 if train else 7)
        im[np.where(true_border)] = (0, 153, 255)
        im[np.where(pred_border)] = (255, 153, 0)
        Image.fromarray(im).save(os.path.join(path, ("train_%i.png" if train else "test_%i.png") % i))
    return ims

def train(args: JobDescription):
    log("Training with", pformat(args.todict()))

    config = TrainConfig(
        splits=args.splits,
        num_models=args.num_models,
        lr=args.lr,
        batches=args.batches,
        batch_size=args.batch_size,
        train_control_prob=args.train_control_prob,
        dropout=args.dropout,
        pretrain_path=args.base_model,
        pretrain=not job.no_pretraining,
    )
    config.save(os.path.join(args.location, "tumor-segmentation-config"))

    control_files, patient_files, extra_patient_files = get_data_files()
    images, segmentations = load_data(control_files, patient_files, extra_patient_files)

    all_results: list[TrainResults] = list()

    for split in range(config.splits):

        log.section("Split %i" % split)
        results = TrainResults.empty(config)
        location = os.path.join(args.location, "split_%i" % split)
        os.makedirs(location, exist_ok="Tue")
        train_images, train_segmentations, test_images, test_segmentations = split_train_test(images, segmentations, config, len(control_files), len(extra_patient_files), split)

        log(
            f"{len(train_images) = :,}",
            f"{len(test_images) = :,}",
            f"{len(control_files) = :,}",
            f"{len(patient_files) = :,}",
            f"{len(extra_patient_files) = :,}",
        )

        augmentations = None if args.no_augment else get_augmentation_pipeline(args.augment_prob)
        train_dataloader = dataloader_(config, train_images, train_segmentations, augmentations=augmentations, n_control=len(control_files))
        test_dataloader = dataloader_(config, test_images, test_segmentations, is_test=True)

        models: list[TumorBoi] = list()
        optimizers = list()
        schedulers = list()
        for i in range(config.num_models):
            model = TumorBoi(config).to(device)
            if not config.pretrain:
                model.mask2former.init_weights()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
            scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_prop * config.batches), config.batches)
            models.append(model)
            optimizers.append(optimizer)
            schedulers.append(scheduler)

        for i in range(config.batches):
            for j, model in enumerate(models):
                ims, segs = next(train_dataloader)
                out = model(ims, segs)

                pred_segs = model.out_to_segs(out)
                dice_score = dice(segs, pred_segs)
                log(
                    "Train %i, %i: %.2f" % (i, j, out.loss.item()),
                    "              %.2f" % dice_score,
                )
                results.train_loss[j].append(out.loss.item())
                results.train_dice[j].append(dice_score)

                out.loss.backward()
                optimizers[j].step()
                optimizers[j].zero_grad()
                schedulers[j].step()
            if (i % args.val_every == 0 or i == (config.batches - 1)) and config.splits > 1:
                plot_samples(location, ims[:5], segs[:5], pred_segs[:5], train=True)
                ims, segs = next(test_dataloader)
                all_pred_segs = [[] for _ in range(len(ims))]
                for j, model in enumerate(models):
                    img_idx = 0
                    model.eval()
                    with torch.inference_mode():
                        out = model(ims, segs)
                    pred_segs = model.out_to_segs(out)
                    for seg in pred_segs:
                        all_pred_segs[img_idx].append(seg)
                        img_idx += 1
                    dice_score = dice(segs, pred_segs)
                    log(
                        "TEST %i, %i: %.2f" % (i, j, out.loss.item()),
                        "             %.2f" % dice_score,
                    )
                    results.test_loss[j].append(out.loss.item())
                    results.test_dice[j].append(dice_score)
                    model.train()

                pred_segs = vote(all_pred_segs)
                dice_score = dice(segs, pred_segs)
                results.ensemble_dice.append(dice_score)
                results.test_batches.append(i)
                plot_samples(location, ims[:5], segs[:5], pred_segs[:5], train=False)

        log.section("Saving")
        results.save(os.path.join(os.path.join(location, "tumor-segmentation-results")))
        all_results.append(results)
        for i, model in enumerate(models):
            torch.save(model.state_dict(), os.path.join(location, "tumor_model_%i.pt" % i))

        plot(location, results, config)

    results = TrainResults.mean(*all_results)
    results.save(args.location)
    plot(args.location, results, config)

if __name__ == "__main__":
    parser = Parser(
        Option("base-model", default="facebook/mask2former-swin-small-ade-semantic"),
        Option("splits", default=5),
        Option("train-control-prob", default=0.5),
        Option("num-models", default=1),
        Option("lr", default=5e-5),
        Option("batches", default=4000),
        Option("batch-size", default=2),
        Option("val-every", default=20),
        Option("augment-prob", default=0.2),
        Option("warmup-prop", default=0.06),
        Option("dropout", default=0.0),
        Flag("no-augment"),
        Flag("no-pretraining"),
        multiple_jobs=True,
    )

    jobs = parser.parse_args()
    for job in jobs:
        log.configure(os.path.join(job.location, "tumor-segmentation-train.log"))
        log.log_repo()
        log.section(f"Starting {job.name}")
        with log.log_errors:
            train(job)
