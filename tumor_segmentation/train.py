import torch
from tqdm import tqdm

from tumor_segmentation import TrainConfig
from tumor_segmentation.data import get_data_files, load_data, split_train_test, dataloader as dataloader_
from tumor_segmentation.model import TumorBoi

config = TrainConfig()
model = TumorBoi(config)

control_files, patient_files = get_data_files()
images, segmentations = load_data(control_files, patient_files)
train_images, train_segmentations, test_images, test_segmentations = split_train_test(images, segmentations)

train_dataloader = dataloader_(config, train_images, train_segmentations)
test_dataloader = dataloader_(config, test_images, test_segmentations)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

for i in range(config.batches):
    if i % 10 == 0:
        ims, segs = next(test_dataloader)
        with torch.inference_mode():
            out = model(ims, segs)
        print("Test %i: %.2f" % (i, out.loss))
    ims, segs = next(train_dataloader)
    out = model(ims, segs)
    print("Train %i: %.2f" % (i, out.loss))
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.save(model.state_dict(), "tumor_segmentation/tumor_model.pt")
