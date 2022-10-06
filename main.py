import torch.nn.functional as F
import os
from constants import DatasetSplit
import torch
from model import PointNetCls
from model_net_dataset import ModelNetDataset
import torch.optim as optim


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

    log_interval = len(train_loader) // 10

    for batch_idx, (points, target) in enumerate(train_loader):

        # points.shape = (num_batch, 3 (num_dims), num_pts)
        optimizer.zero_grad()
        points = points.cuda()
        target = target.cuda()
        points = points.transpose(2, 1)
        pred, _, _ = model(points)  # forward
        loss = F.nll_loss(pred, target)  # = sum_k(-t_k * log(y_k))
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(points)}/{num_train_samples} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.data.item():.6f}")

            # Log train/loss to TensorBoard at every iteration
            n_iter = (epoch - 1) * num_train_samples + batch_idx * len(points)
            if writer is not None:
                writer.add_scalar('train/loss', loss.data.item(), n_iter)

    # # Log model parameters to TensorBoard at every epoch
    # if writer is not None:
    #     for name, param in model.named_parameters():
    #         layer, attr = os.path.splitext(name)
    #         attr = attr[1:]
    #         writer.add_histogram('{}/{}'.format(layer, attr), param.clone().cpu().data.numpy(), n_iter)


# evaluate model on the given data loader
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
        pred, _, _ = model(points)  # forward
        test_loss += F.nll_loss(pred, target, size_average=False).data.item()
        pred = pred.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    num_samples = len(data_loader.dataset)
    test_loss /= num_samples
    test_accuracy = 100. * correct / num_samples
    print(f"{dataset_split.value} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{num_samples} ({test_accuracy:.2f}%)\n")

    if n_iter is not None and writer is not None:
        writer.add_scalar(f'{dataset_split.value}/loss', test_loss, n_iter)
        writer.add_scalar(f'{dataset_split.value}/accuracy', test_accuracy, n_iter)


from tensorboardX import SummaryWriter
from datetime import datetime
import os

writer = SummaryWriter(log_dir=os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S')))


pos_objects = ["bed", "chair", "desk", "monitor", "table"]
num_classes = len(pos_objects) + 1

batch_size = 32
num_workers = 4
num_epcohs = 50

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

# Setup the network
model = PointNetCls(k=num_classes)
model.cuda()

# Visualize network as a graph on TensorBoard
# input_tensor = torch.Tensor(3,3,1024)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Start training
for epoch in range(1, num_epcohs + 1):
    train(
        model=model,
        epoch=epoch,
        train_loader=train_loader,
        optimizer=optimizer,
        writer=writer
    )

    evaluate(
        model=model,
        dataset_split=DatasetSplit.VAL,
        data_loader=val_loader,
        writer=writer,
        n_iter=epoch * len(train_set))

evaluate(
    model=model,
    dataset_split=DatasetSplit.TEST,
    data_loader=test_loader,
    writer=writer)

# Close TensorBoardX summary writer
writer.close()