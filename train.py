from constants import DatasetSplit
import torch
from model import PointNetCls
from model_net_dataset import ModelNetDataset
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
import os
from model_training import train, evaluate
import argparse
from typing import List
from utils import random_utils

def main(pos_objects:List[str]):
    """Train a model that identifies object"""
    random_utils.set_random_seed()

    run_dir = os.path.join('runs', datetime.now().strftime('%b-%H-%M-%S'))
    writer = SummaryWriter(log_dir=run_dir)

    num_classes = len(pos_objects) + 1

    batch_size = 64
    num_workers = 8
    num_epochs = 150

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
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    # endregion data loaders

    # region model training

    # Setup PointNet-based classification model
    model = PointNetCls(k=num_classes)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,110,130], gamma=0.5)

    # Start training
    for epoch in range(1, num_epochs + 1):
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
        data_loader=test_loader
    )

    model_file = os.path.join(run_dir, "model_final.pth")
    torch.save(model.state_dict(), model_file)


if __name__ == "__main__":
    pos_objects = ["bed", "chair", "desk", "monitor", "table"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--pos-objects", nargs="*", default=pos_objects)
    args = parser.parse_args()

    main(
        pos_objects=args.pos_objects,
    )