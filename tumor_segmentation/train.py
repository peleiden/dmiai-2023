import os
from pelutils import JobDescription, Option, Parser, log, Flag
import matplotlib.pyplot as plt
import pelutils.ds.plots as plots
import torch
from pelutils import log
from transformers import get_linear_schedule_with_warmup

from tumor_segmentation import device, TrainConfig, TrainResults
from tumor_segmentation.data import dice, get_data_files, load_data, split_train_test, dataloader as dataloader_, vote, get_augmentation_pipeline
from tumor_segmentation.model import UNETTTT, TumorBoi


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

def train(args: JobDescription):
    log("Training with", args)

    config = TrainConfig(
        train_test_split=args.train_split,
        num_models=args.num_models,
        lr=args.lr,
        batches=args.batches,
        batch_size=args.batch_size,
        train_control_prob=args.train_control_prob,
        dropout=args.dropout,
    )
    config.save(os.path.join(args.location, "tumor-segmentation-config"))

    results = TrainResults.empty(config)

    control_files, patient_files, extra_patient_files = get_data_files()
    images, segmentations = load_data(control_files, patient_files, extra_patient_files)
    train_images, train_segmentations, test_images, test_segmentations = split_train_test(images, segmentations, config, len(control_files), len(extra_patient_files))

    augmentations = None if args.no_augment else get_augmentation_pipeline(args.augment_prob)
    train_dataloader = dataloader_(config, train_images, train_segmentations, augmentations=augmentations, n_control=len(control_files))
    test_dataloader = dataloader_(config, test_images, test_segmentations, is_test=True)

    models: list[TumorBoi] = list()
    optimizers = list()
    schedulers = list()
    for i in range(config.num_models):
        model = (UNETTTT if args.unet else TumorBoi)(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_prop * config.batches), config.batches)
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    for i in range(config.batches):
        for j, model in enumerate(models):
            ims, segs = next(train_dataloader)
            out = model(ims, segs)

            pred_segs = model.out_to_segs(out, [seg.shape for seg in segs])
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
        if i % args.val_every == 0 or i == (config.batches - 1):
            ims, segs = next(test_dataloader)
            all_pred_segs = [[] for _ in range(len(ims))]
            for j, model in enumerate(models):
                img_idx = 0
                model.eval()
                with torch.inference_mode():
                    out = model(ims, segs)
                pred_segs = model.out_to_segs(out, [seg.shape for seg in segs])
                for seg in pred_segs:
                    all_pred_segs[img_idx].append(seg)
                    img_idx += 1
                dice_score = dice(segs, pred_segs)
                log(
                    "Test %i, %i: %.2f" % (i, j, out.loss.item()),
                    "             %.2f" % dice_score,
                )
                results.test_loss[j].append(out.loss.item())
                results.test_dice[j].append(dice_score)
                model.train()

            pred_segs = vote(all_pred_segs)
            dice_score = dice(segs, pred_segs)
            results.ensemble_dice.append(dice_score)
            results.test_batches.append(i)

    log.section("Saving")
    results.save(os.path.join(os.path.join(args.location, "tumor-segmentation-results")))
    for i, model in enumerate(models):
        torch.save(model.state_dict(), os.path.join(args.location, "tumor_model_%i.pt" % i))

    plot(args.location, results, config)

if __name__ == "__main__":
    parser = Parser(
        Option("base-model", default="facebook/mask2former-swin-small-ade-semantic"),
        Option("train-split", default=0.80), # prop of tumor images to train on
        Option("train-control-prob", default=0.5),
        Option("num-models", default=1),
        Option("lr", default=5e-5),
        Option("batches", default=500),
        Option("batch-size", default=32),
        Option("val-every", default=10),
        Option("augment-prob", default=0.2),
        Option("warmup-prop", default=0.06),
        Option("dropout", default=0.0),
        Flag("no-augment"),
        Flag("unet"),
        multiple_jobs=True,
    )

    jobs = parser.parse_args()
    for job in jobs:
        log.configure(
            os.path.join(job.location, "tumor-segmentation-train.log"),
            append=True,
        )
        log.log_repo()
        log.section(f"Starting {job.name}")
        with log.log_errors:
            train(job)
