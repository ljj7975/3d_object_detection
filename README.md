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

### Training
`python train.py` will start the training and save the trained model under `runs/<date_time>` along with tensorboard logs. 

### Evaluation
`python evaluate.py --model-path <location of the .pth file>` will run the evaluation using the trained model 
