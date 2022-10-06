from constants import DatasetSplit
import torch
from model import PointNetCls
from model_net_dataset import ModelNetDataset
import argparse
from model_training import evaluate
from typing import List
from utils import random_utils

def main(model_path:str, pos_objects:List[str]):
    """Load pretrained model and evaluate on test set"""
    random_utils.set_random_seed()

    num_classes = len(pos_objects) + 1

    batch_size = 32
    num_workers = 8

    # region data loaders
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

    # Load pre-trained model
    model = PointNetCls(k=num_classes)
    model.cuda()
    model.load_state_dict(torch.load(model_path))

    evaluate(
        model=model,
        dataset_split=DatasetSplit.TEST,
        data_loader=test_loader
    )

if __name__ == "__main__":
    pos_objects = ["bed", "chair", "desk", "monitor", "table"]
    model_path = "pretrained/model_final.pth"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=model_path)
    parser.add_argument("--pos-objects", nargs="*", default=pos_objects)
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        pos_objects=args.pos_objects,
    )