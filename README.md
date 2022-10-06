# 3d_object_detection

## Problem Overview
As a research scientist, you are working on the solving the problem of identifying 5 different
objects: “bed”, “chair”, “desk”, “monitor”, and “table” from a given set of 3D point cloud data
collected from an indoor environment.

You are asked to work with an available dataset, ModelNet10, which contains all these objects.
You are tasked with building a predictive model to differentiate between the objects. The
following instructions in the form of subtasks are available to you as a guide for succeeding in
the overall task.

## Instruction

### Downloading dataset
```
wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip ModelNet10.zip
```

### Preparation
```commandline
pip install -r requirements.txt
pip install -r packages.txt
```

### Training
`python train.py` will start the training and save the trained model under `runs/<date_time>` along with tensorboard logs. 

### Evaluation
`python evaluate.py --model-path <location of the .pth file>` will run the evaluation using the trained model.

`python evaluate.py` will run the pre-trained model and report test accuracy.

### File structure
```commandline
.
├── README.md
├── packages.txt
├── requirements.txt
├── constants.py
├── train.py - train a new model
├── evaluate.py - evaulate trained model
├── ModelNet10 - dataset
├── model_net_dataset.py - PyTorch dataset implementation for ModelNet10
├── model_net_dataset_test.py
├── model.py - PointNet implementation
├── model_training.py - train and evaluate function
├── pc_augmentation.py - a set of augmentations applied on train set
├── pc_augmentation_test.py
├── runs - logs and trained models
├── pretrained - trained_models
├── lib
│   └── scene_vis - visualization library
└── utils
    └── vis_utils.py
```

Please note that, even though I have followed subtasks, the final implementation has diverged slightly.

### Additional packages used

I have used `open3d` for loading `.off` file.

I also used `vtk` for visualization along with a [wrapper](https://github.com/kujason/scene_vis)

The full list can be found in [packages.txt](packages.txt) and [requirements.txt](requirements.txt)

## Discussions

### Model selection

There are many models available for 3d object detection.
However, PointNet (developed in 2017) is known in the domain as it can efficiently extract meaningful information from 3d data by processing each point independently.
Later on, authors have improved PointNet further by capturing local structure better (PointNet++)

Since then, PointNet++ has became base feature extractor for many models
* VoteNet: Hough voting applied to the seed points collected using PointNet++ 
* PVN3D: Keypoints voting network for 6DoF Pose estimation; uses RGB features along with PointNet++ features

There also exist researchers working on non-PointNet models.
One notable research is from Samsung AI called FCAF3D which attempts to process 3d data without anchor. 

In this assignment, I simply decide to use PointNet for simplicity.
I have only copied the [model implementation](https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py).
All the other components are implemented by myself.


### Data augmentation

Data augmentation plays a critical role in model generalization.
Without data augmentation, the model will likely overfit to the train set and introduce high variance.

Here are the augmentations I have applied, implementation can be found [here](pc_augmentation.py):
* normalization
* random rotation
* random translation
* gaussian noise for each point
* mixing vertices with the randomly selected points


### Question 1

The final model `pretrained/model_final.pth` reports 79.69% (561/704) on validation set (20% of training set)
and 74.35% (516/694) on test set. The log can be found [here](training_log.md)

There are number of techniques that I can apply to improve the numbers

#### Better model architecture
The naive PointNet is fairly old. The recent models performs much better in general

#### Different weights for each class
I found that there are fewer samples for desk. 
I didn't explicitly log accuracy per object but if the accuracy for desk is particularly lower than other ones, I can weight the desk samples during training so the model is trained with balanced dataset.

Additionally, it's possible that false positive rate might be high (detecting negative samples as one of the positive object).
In this case, I'd add more negative samples to the training set. I can apply augmentations more aggressively on these samples.

Focal loss can possibly used in this case.

#### Increasing the number of points
I am currently using 1024 points as instructed by the assignment.
However, 1024 points might be too small to distinguish the objects.

#### Early Stopping
I train the model for a fixed iteration, but I can save the model that reports the highest number for validation set.

#### Hyperparameter tuning
There are many parameters that can be tuned. I think I could've applied cross validation to search for the optional sets.

#### Ensemble technique
Since the model achieves accuracy > 0.5, we can train multiple models using different training set and aggregate the results for the final prediction.

### Question 2

Throughout the training, the model learns features that helps identifying the object from 3d data.
As a result, activations for the same class will be similar while notably different from activations for other classes (especially in the layers at the end).
We can possibly compare these activations and see if the new class can be identified without retraining the model.

If we look at these activations as distributions, we need a metric that measures the distance between two distributions.
However, we cannot assume anything about the underlying distributions, so I think we can try Kruskal-Wallis Test.

In fact, you can possibly consider this process as measuring the distance between two points in x-dims, where x is the number of activations.
In this case, you can simply calculate norm (e.g. mean squared error or mean absolute error).

To further increase the robustness in this process, we can use a centroid (or average) point for each class collected from validation set.
