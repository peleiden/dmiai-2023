import cv2
import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
import torch
from pelutils import log
from transformers import get_linear_schedule_with_warmup

from tumor_segmentation import device, TrainConfig, TrainResults
from tumor_segmentation.data import dice, get_data_files, load_data, split_train_test, dataloader as dataloader_, vote
from tumor_segmentation.model import TumorBoi


log.configure("train-tumor.log")

config = TrainConfig()
config.save("tumor_segmentation")
results = TrainResults.empty(config)

control_files, patient_files = get_data_files()
images, segmentations = load_data(control_files, patient_files)
train_images, train_segmentations, test_images, test_segmentations = split_train_test(images, segmentations, config)

train_dataloader = dataloader_(config, train_images, train_segmentations)
test_dataloader = dataloader_(config, test_images, test_segmentations)

models: list[TumorBoi] = list()
optimizers = list()
schedulers = list()
for i in range(config.num_models):
    model = TumorBoi(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * config.batches), config.batches)
    models.append(model)
    optimizers.append(optimizer)
    schedulers.append(scheduler)

def out_to_seg(out) -> np.ndarray:
    segs = model.processor.post_process_semantic_segmentation(out)
    segs = [seg.cpu().numpy().astype(np.uint8) for seg in segs]
    segs = np.array([cv2.resize(seg, (400, 991)).astype(bool) for seg in segs])
    return segs

for i in range(config.batches):
    if i % 10 == 0:
        ims, segs = next(test_dataloader)
        all_pred_segs = list()
        for j, model in enumerate(models):
            model.eval()
            with torch.inference_mode():
                out = model(ims, segs)
            pred_segs = out_to_seg(out)
            all_pred_segs.append(pred_segs)
            dice_score = dice(segs, pred_segs)
            log(
                "Test %i, %i: %.2f" % (i, j, out.loss.item()),
                "             %.2f" % dice_score,
            )
            results.test_loss[j].append(out.loss.item())
            results.test_dice[j].append(dice_score)
            model.train()

        all_pred_segs = np.array(all_pred_segs)
        pred_segs = vote(all_pred_segs)
        dice_score = dice(segs, all_pred_segs)
        results.ensemble_dice.append(dice_score)
        results.test_batches.append(i)

    for j, model in enumerate(models):
        ims, segs = next(train_dataloader)
        out = model(ims, segs)
        log("Train %i, %i: %.2f" % (i, j, out.loss))

        pred_segs = out_to_seg(out)
        dice_score = dice(segs, pred_segs)

        results.train_loss[j].append(out.loss.item())
        results.train_dice[j].append(dice_score)

        out.loss.backward()
        optimizers[j].step()
        optimizers[j].zero_grad()
        schedulers[j].step()

log.section("Saving")
results.save("tumor_segmentation")
for i, model in enumerate(models):
    torch.save(model.state_dict(), "tumor_segmentation/tumor_model_%i.pt" % i)

log.section("Plotting")
with plots.Figure("tumor_segmentation/train.png", figsize=(20, 10), fontsize=18):
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
