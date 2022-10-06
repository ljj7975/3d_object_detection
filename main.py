import torch.nn.functional as F
from constants import DatasetSplit
import torch
from model import PointNetCls
from model_net_dataset import ModelNetDataset
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
import os

# train single epoch
def train(
    model,
    epoch,
    train_loader,
    optimizer,
    writer = None,
):
    model.train()
    num_train_samples = len(train_loader.dataset)
    log_interval = len(train_loader) // 5

    for batch_idx, (points, target) in enumerate(train_loader):
        optimizer.zero_grad()
        points = points.cuda()
        target = target.cuda()
        points = points.transpose(2, 1) # points.shape needs to be (num_batch, 3 (num_dims), num_pts)
        output, _, _ = model(points)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(points)}/{num_train_samples} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.data.item():.6f}")

            # Log train/loss to TensorBoard
            n_iter = (epoch - 1) * num_train_samples + batch_idx * len(points)
            if writer is not None:
                writer.add_scalar('train/loss', loss.data.item(), n_iter)


# evaluate model
def evaluate(
    model,
    dataset_split,
    data_loader,
    writer=None,
    n_iter = None
):
    model.eval()
    test_loss = 0
    correct = 0

    for points, target in data_loader:
        points = points.cuda()
        target = target.cuda()
        points = points.transpose(2, 1)
        output, _, _ = model(points)
        test_loss += F.nll_loss(output, target, size_average=False).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    num_samples = len(data_loader.dataset)
    test_loss /= num_samples
    test_accuracy = 100. * correct / num_samples
    print(f"{dataset_split.value} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{num_samples} ({test_accuracy:.2f}%)\n")

    # Log val/loss & val/accuracy
    if n_iter is not None and writer is not None:
        writer.add_scalar(f'{dataset_split.value}/loss', test_loss, n_iter)
        writer.add_scalar(f'{dataset_split.value}/accuracy', test_accuracy, n_iter)


writer = SummaryWriter(log_dir=os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S')))

pos_objects = ["bed", "chair", "desk", "monitor", "table"]
num_classes = len(pos_objects) + 1

batch_size = 32
num_workers = 8
num_epcohs = 200

# region data loaders
train_set = ModelNetDataset(
    dataset_split = DatasetSplit.TRAIN,
    pos_objects=pos_objects
)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)

val_set = ModelNetDataset(
    dataset_split = DatasetSplit.VAL,
    pos_objects=pos_objects
)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)

test_set = ModelNetDataset(
    dataset_split = DatasetSplit.TEST,
    pos_objects=pos_objects
)
test_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)

# endregion data loaders

# region model training

# Setup PointNet-based classification model
model = PointNetCls(k=num_classes)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)

# Start training
for epoch in range(1, num_epcohs + 1):
    scheduler.step()
    train(
        model=model,
        epoch=epoch,
        train_loader=train_loader,
        optimizer=optimizer,
        writer=writer
    )

    n_iter = epoch * len(train_set)
    writer.add_scalar('lr', scheduler.get_lr(), n_iter)

    evaluate(
        model=model,
        dataset_split=DatasetSplit.VAL,
        data_loader=val_loader,
        writer=writer,
        n_iter=n_iter)

writer.close()

# endregion model training

evaluate(
    model=model,
    dataset_split=DatasetSplit.TEST,
    data_loader=test_loader,
    writer=writer
)
