```commandline
(3d) brandon@ku:~/personal/lg/3d_object_detection$ python train.py 
counts per label
	0: 412
	1: 711
	2: 160
	3: 372
	4: 313
	5: 843
counts per label
	0: 103
	1: 178
	2: 40
	3: 93
	4: 79
	5: 211
counts per label
	0: 100
	1: 100
	2: 86
	3: 100
	4: 100
	5: 208
/home/brandon/.virtualenvs/3d/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:134: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
Train Epoch: 1 [0/2811 (0%)]	Loss: 1.936601
Train Epoch: 1 [512/2811 (18%)]	Loss: 1.901343
Train Epoch: 1 [1024/2811 (36%)]	Loss: 1.674450
Train Epoch: 1 [1536/2811 (55%)]	Loss: 1.653273
Train Epoch: 1 [2048/2811 (73%)]	Loss: 1.534730
Train Epoch: 1 [2560/2811 (91%)]	Loss: 1.517094
/home/brandon/.virtualenvs/3d/lib/python3.6/site-packages/torch/optim/lr_scheduler.py:417: UserWarning: To get the last learning rate computed by the scheduler, please use `get_last_lr()`.
  "please use `get_last_lr()`.", UserWarning)
/home/brandon/.virtualenvs/3d/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))
val set: Average loss: 1.7120, Accuracy: 211/704 (29.97%)

Train Epoch: 2 [0/2811 (0%)]	Loss: 1.759399
Train Epoch: 2 [512/2811 (18%)]	Loss: 1.671866
Train Epoch: 2 [1024/2811 (36%)]	Loss: 1.494798
Train Epoch: 2 [1536/2811 (55%)]	Loss: 1.526361
Train Epoch: 2 [2048/2811 (73%)]	Loss: 1.601655
Train Epoch: 2 [2560/2811 (91%)]	Loss: 1.254225
val set: Average loss: 1.6623, Accuracy: 222/704 (31.53%)

Train Epoch: 3 [0/2811 (0%)]	Loss: 1.334906
Train Epoch: 3 [512/2811 (18%)]	Loss: 1.316578
Train Epoch: 3 [1024/2811 (36%)]	Loss: 1.368685
Train Epoch: 3 [1536/2811 (55%)]	Loss: 1.335935
Train Epoch: 3 [2048/2811 (73%)]	Loss: 1.164773
Train Epoch: 3 [2560/2811 (91%)]	Loss: 1.351310
val set: Average loss: 3.0128, Accuracy: 117/704 (16.62%)

Train Epoch: 4 [0/2811 (0%)]	Loss: 1.389427
Train Epoch: 4 [512/2811 (18%)]	Loss: 1.265353
Train Epoch: 4 [1024/2811 (36%)]	Loss: 1.178569
Train Epoch: 4 [1536/2811 (55%)]	Loss: 1.219124
Train Epoch: 4 [2048/2811 (73%)]	Loss: 1.147322
Train Epoch: 4 [2560/2811 (91%)]	Loss: 1.313936
val set: Average loss: 1.6143, Accuracy: 275/704 (39.06%)

Train Epoch: 5 [0/2811 (0%)]	Loss: 1.227370
Train Epoch: 5 [512/2811 (18%)]	Loss: 1.089920
Train Epoch: 5 [1024/2811 (36%)]	Loss: 1.196626
Train Epoch: 5 [1536/2811 (55%)]	Loss: 1.445666
Train Epoch: 5 [2048/2811 (73%)]	Loss: 1.332975
Train Epoch: 5 [2560/2811 (91%)]	Loss: 1.025023
val set: Average loss: 1.0275, Accuracy: 422/704 (59.94%)

Train Epoch: 6 [0/2811 (0%)]	Loss: 1.090976
Train Epoch: 6 [512/2811 (18%)]	Loss: 1.313897
Train Epoch: 6 [1024/2811 (36%)]	Loss: 0.839542
Train Epoch: 6 [1536/2811 (55%)]	Loss: 1.077432
Train Epoch: 6 [2048/2811 (73%)]	Loss: 0.875293
Train Epoch: 6 [2560/2811 (91%)]	Loss: 1.185459
val set: Average loss: 0.9369, Accuracy: 455/704 (64.63%)

Train Epoch: 7 [0/2811 (0%)]	Loss: 0.986526
Train Epoch: 7 [512/2811 (18%)]	Loss: 1.018460
Train Epoch: 7 [1024/2811 (36%)]	Loss: 1.124170
Train Epoch: 7 [1536/2811 (55%)]	Loss: 1.093157
Train Epoch: 7 [2048/2811 (73%)]	Loss: 1.165814
Train Epoch: 7 [2560/2811 (91%)]	Loss: 1.200251
val set: Average loss: 1.2639, Accuracy: 372/704 (52.84%)

Train Epoch: 8 [0/2811 (0%)]	Loss: 1.129051
Train Epoch: 8 [512/2811 (18%)]	Loss: 0.989446
Train Epoch: 8 [1024/2811 (36%)]	Loss: 1.054979
Train Epoch: 8 [1536/2811 (55%)]	Loss: 1.129062
Train Epoch: 8 [2048/2811 (73%)]	Loss: 1.055911
Train Epoch: 8 [2560/2811 (91%)]	Loss: 0.954433
val set: Average loss: 0.8859, Accuracy: 483/704 (68.61%)

Train Epoch: 9 [0/2811 (0%)]	Loss: 0.802514
Train Epoch: 9 [512/2811 (18%)]	Loss: 0.872757
Train Epoch: 9 [1024/2811 (36%)]	Loss: 0.832937
Train Epoch: 9 [1536/2811 (55%)]	Loss: 1.227250
Train Epoch: 9 [2048/2811 (73%)]	Loss: 0.884163
Train Epoch: 9 [2560/2811 (91%)]	Loss: 1.118471
val set: Average loss: 1.0243, Accuracy: 455/704 (64.63%)

Train Epoch: 10 [0/2811 (0%)]	Loss: 0.923786
Train Epoch: 10 [512/2811 (18%)]	Loss: 1.151044
Train Epoch: 10 [1024/2811 (36%)]	Loss: 0.953595
Train Epoch: 10 [1536/2811 (55%)]	Loss: 0.917322
Train Epoch: 10 [2048/2811 (73%)]	Loss: 1.052207
Train Epoch: 10 [2560/2811 (91%)]	Loss: 0.757051
val set: Average loss: 1.0121, Accuracy: 460/704 (65.34%)

Train Epoch: 11 [0/2811 (0%)]	Loss: 0.854337
Train Epoch: 11 [512/2811 (18%)]	Loss: 1.016633
Train Epoch: 11 [1024/2811 (36%)]	Loss: 0.633532
Train Epoch: 11 [1536/2811 (55%)]	Loss: 0.866740
Train Epoch: 11 [2048/2811 (73%)]	Loss: 0.943299
Train Epoch: 11 [2560/2811 (91%)]	Loss: 0.878377
val set: Average loss: 1.0646, Accuracy: 436/704 (61.93%)

Train Epoch: 12 [0/2811 (0%)]	Loss: 0.846725
Train Epoch: 12 [512/2811 (18%)]	Loss: 0.912335
Train Epoch: 12 [1024/2811 (36%)]	Loss: 0.853503
Train Epoch: 12 [1536/2811 (55%)]	Loss: 0.910126
Train Epoch: 12 [2048/2811 (73%)]	Loss: 0.920468
Train Epoch: 12 [2560/2811 (91%)]	Loss: 0.781567
val set: Average loss: 1.2197, Accuracy: 394/704 (55.97%)

Train Epoch: 13 [0/2811 (0%)]	Loss: 0.901123
Train Epoch: 13 [512/2811 (18%)]	Loss: 0.978241
Train Epoch: 13 [1024/2811 (36%)]	Loss: 1.109971
Train Epoch: 13 [1536/2811 (55%)]	Loss: 0.837901
Train Epoch: 13 [2048/2811 (73%)]	Loss: 1.018789
Train Epoch: 13 [2560/2811 (91%)]	Loss: 0.756431
val set: Average loss: 0.9575, Accuracy: 466/704 (66.19%)

Train Epoch: 14 [0/2811 (0%)]	Loss: 1.019982
Train Epoch: 14 [512/2811 (18%)]	Loss: 0.910600
Train Epoch: 14 [1024/2811 (36%)]	Loss: 0.898768
Train Epoch: 14 [1536/2811 (55%)]	Loss: 1.065112
Train Epoch: 14 [2048/2811 (73%)]	Loss: 1.181432
Train Epoch: 14 [2560/2811 (91%)]	Loss: 0.826227
val set: Average loss: 1.0045, Accuracy: 421/704 (59.80%)

Train Epoch: 15 [0/2811 (0%)]	Loss: 1.134452
Train Epoch: 15 [512/2811 (18%)]	Loss: 1.066001
Train Epoch: 15 [1024/2811 (36%)]	Loss: 0.700933
Train Epoch: 15 [1536/2811 (55%)]	Loss: 0.718690
Train Epoch: 15 [2048/2811 (73%)]	Loss: 0.949717
Train Epoch: 15 [2560/2811 (91%)]	Loss: 0.871740
val set: Average loss: 0.9753, Accuracy: 447/704 (63.49%)

Train Epoch: 16 [0/2811 (0%)]	Loss: 0.906023
Train Epoch: 16 [512/2811 (18%)]	Loss: 0.914828
Train Epoch: 16 [1024/2811 (36%)]	Loss: 0.854286
Train Epoch: 16 [1536/2811 (55%)]	Loss: 0.853927
Train Epoch: 16 [2048/2811 (73%)]	Loss: 0.708314
Train Epoch: 16 [2560/2811 (91%)]	Loss: 0.755695
val set: Average loss: 0.7540, Accuracy: 516/704 (73.30%)

Train Epoch: 17 [0/2811 (0%)]	Loss: 0.897274
Train Epoch: 17 [512/2811 (18%)]	Loss: 0.824205
Train Epoch: 17 [1024/2811 (36%)]	Loss: 0.815137
Train Epoch: 17 [1536/2811 (55%)]	Loss: 0.970853
Train Epoch: 17 [2048/2811 (73%)]	Loss: 0.593686
Train Epoch: 17 [2560/2811 (91%)]	Loss: 0.820157
val set: Average loss: 1.0436, Accuracy: 419/704 (59.52%)

Train Epoch: 18 [0/2811 (0%)]	Loss: 0.697735
Train Epoch: 18 [512/2811 (18%)]	Loss: 1.018363
Train Epoch: 18 [1024/2811 (36%)]	Loss: 1.047484
Train Epoch: 18 [1536/2811 (55%)]	Loss: 1.017510
Train Epoch: 18 [2048/2811 (73%)]	Loss: 0.758186
Train Epoch: 18 [2560/2811 (91%)]	Loss: 0.930936
val set: Average loss: 0.9122, Accuracy: 467/704 (66.34%)

Train Epoch: 19 [0/2811 (0%)]	Loss: 0.975779
Train Epoch: 19 [512/2811 (18%)]	Loss: 0.839629
Train Epoch: 19 [1024/2811 (36%)]	Loss: 0.923791
Train Epoch: 19 [1536/2811 (55%)]	Loss: 0.912717
Train Epoch: 19 [2048/2811 (73%)]	Loss: 0.803613
Train Epoch: 19 [2560/2811 (91%)]	Loss: 0.752404
val set: Average loss: 1.2319, Accuracy: 409/704 (58.10%)

Train Epoch: 20 [0/2811 (0%)]	Loss: 0.666852
Train Epoch: 20 [512/2811 (18%)]	Loss: 0.791711
Train Epoch: 20 [1024/2811 (36%)]	Loss: 0.848599
Train Epoch: 20 [1536/2811 (55%)]	Loss: 0.816509
Train Epoch: 20 [2048/2811 (73%)]	Loss: 0.677251
Train Epoch: 20 [2560/2811 (91%)]	Loss: 0.854933
val set: Average loss: 0.8147, Accuracy: 491/704 (69.74%)

Train Epoch: 21 [0/2811 (0%)]	Loss: 0.865161
Train Epoch: 21 [512/2811 (18%)]	Loss: 0.819862
Train Epoch: 21 [1024/2811 (36%)]	Loss: 0.587365
Train Epoch: 21 [1536/2811 (55%)]	Loss: 0.778167
Train Epoch: 21 [2048/2811 (73%)]	Loss: 0.765628
Train Epoch: 21 [2560/2811 (91%)]	Loss: 0.742653
val set: Average loss: 0.9678, Accuracy: 446/704 (63.35%)

Train Epoch: 22 [0/2811 (0%)]	Loss: 0.983192
Train Epoch: 22 [512/2811 (18%)]	Loss: 0.871932
Train Epoch: 22 [1024/2811 (36%)]	Loss: 0.891202
Train Epoch: 22 [1536/2811 (55%)]	Loss: 0.616132
Train Epoch: 22 [2048/2811 (73%)]	Loss: 0.685746
Train Epoch: 22 [2560/2811 (91%)]	Loss: 0.812097
val set: Average loss: 0.8408, Accuracy: 484/704 (68.75%)

Train Epoch: 23 [0/2811 (0%)]	Loss: 0.938486
Train Epoch: 23 [512/2811 (18%)]	Loss: 0.648738
Train Epoch: 23 [1024/2811 (36%)]	Loss: 0.770342
Train Epoch: 23 [1536/2811 (55%)]	Loss: 0.773944
Train Epoch: 23 [2048/2811 (73%)]	Loss: 0.687439
Train Epoch: 23 [2560/2811 (91%)]	Loss: 0.805992
val set: Average loss: 0.7322, Accuracy: 511/704 (72.59%)

Train Epoch: 24 [0/2811 (0%)]	Loss: 0.853562
Train Epoch: 24 [512/2811 (18%)]	Loss: 0.993683
Train Epoch: 24 [1024/2811 (36%)]	Loss: 0.704504
Train Epoch: 24 [1536/2811 (55%)]	Loss: 0.734907
Train Epoch: 24 [2048/2811 (73%)]	Loss: 0.748170
Train Epoch: 24 [2560/2811 (91%)]	Loss: 0.929285
val set: Average loss: 0.9511, Accuracy: 456/704 (64.77%)

Train Epoch: 25 [0/2811 (0%)]	Loss: 0.668297
Train Epoch: 25 [512/2811 (18%)]	Loss: 0.714477
Train Epoch: 25 [1024/2811 (36%)]	Loss: 0.669620
Train Epoch: 25 [1536/2811 (55%)]	Loss: 0.879092
Train Epoch: 25 [2048/2811 (73%)]	Loss: 0.997695
Train Epoch: 25 [2560/2811 (91%)]	Loss: 0.795131
val set: Average loss: 0.7239, Accuracy: 530/704 (75.28%)

Train Epoch: 26 [0/2811 (0%)]	Loss: 0.667960
Train Epoch: 26 [512/2811 (18%)]	Loss: 0.643005
Train Epoch: 26 [1024/2811 (36%)]	Loss: 0.743650
Train Epoch: 26 [1536/2811 (55%)]	Loss: 0.785092
Train Epoch: 26 [2048/2811 (73%)]	Loss: 0.845705
Train Epoch: 26 [2560/2811 (91%)]	Loss: 0.648033
val set: Average loss: 0.8329, Accuracy: 508/704 (72.16%)

Train Epoch: 27 [0/2811 (0%)]	Loss: 0.622285
Train Epoch: 27 [512/2811 (18%)]	Loss: 0.678965
Train Epoch: 27 [1024/2811 (36%)]	Loss: 0.706312
Train Epoch: 27 [1536/2811 (55%)]	Loss: 0.839692
Train Epoch: 27 [2048/2811 (73%)]	Loss: 0.512095
Train Epoch: 27 [2560/2811 (91%)]	Loss: 0.756961
val set: Average loss: 0.7800, Accuracy: 503/704 (71.45%)

Train Epoch: 28 [0/2811 (0%)]	Loss: 0.674189
Train Epoch: 28 [512/2811 (18%)]	Loss: 0.761835
Train Epoch: 28 [1024/2811 (36%)]	Loss: 0.611245
Train Epoch: 28 [1536/2811 (55%)]	Loss: 0.758513
Train Epoch: 28 [2048/2811 (73%)]	Loss: 0.899967
Train Epoch: 28 [2560/2811 (91%)]	Loss: 0.886687
val set: Average loss: 0.7576, Accuracy: 517/704 (73.44%)

Train Epoch: 29 [0/2811 (0%)]	Loss: 1.017345
Train Epoch: 29 [512/2811 (18%)]	Loss: 0.898528
Train Epoch: 29 [1024/2811 (36%)]	Loss: 0.917543
Train Epoch: 29 [1536/2811 (55%)]	Loss: 0.842904
Train Epoch: 29 [2048/2811 (73%)]	Loss: 0.764243
Train Epoch: 29 [2560/2811 (91%)]	Loss: 0.638224
val set: Average loss: 0.8288, Accuracy: 479/704 (68.04%)

Train Epoch: 30 [0/2811 (0%)]	Loss: 0.624053
Train Epoch: 30 [512/2811 (18%)]	Loss: 0.697018
Train Epoch: 30 [1024/2811 (36%)]	Loss: 0.773928
Train Epoch: 30 [1536/2811 (55%)]	Loss: 0.725017
Train Epoch: 30 [2048/2811 (73%)]	Loss: 0.631193
Train Epoch: 30 [2560/2811 (91%)]	Loss: 0.624530
val set: Average loss: 0.7960, Accuracy: 499/704 (70.88%)

Train Epoch: 31 [0/2811 (0%)]	Loss: 1.085783
Train Epoch: 31 [512/2811 (18%)]	Loss: 0.637491
Train Epoch: 31 [1024/2811 (36%)]	Loss: 0.747059
Train Epoch: 31 [1536/2811 (55%)]	Loss: 0.776098
Train Epoch: 31 [2048/2811 (73%)]	Loss: 0.883558
Train Epoch: 31 [2560/2811 (91%)]	Loss: 0.737973
val set: Average loss: 0.6747, Accuracy: 529/704 (75.14%)

Train Epoch: 32 [0/2811 (0%)]	Loss: 0.652426
Train Epoch: 32 [512/2811 (18%)]	Loss: 0.776483
Train Epoch: 32 [1024/2811 (36%)]	Loss: 0.928322
Train Epoch: 32 [1536/2811 (55%)]	Loss: 0.651515
Train Epoch: 32 [2048/2811 (73%)]	Loss: 0.593004
Train Epoch: 32 [2560/2811 (91%)]	Loss: 0.773482
val set: Average loss: 0.8487, Accuracy: 496/704 (70.45%)

Train Epoch: 33 [0/2811 (0%)]	Loss: 0.901943
Train Epoch: 33 [512/2811 (18%)]	Loss: 0.698710
Train Epoch: 33 [1024/2811 (36%)]	Loss: 0.708604
Train Epoch: 33 [1536/2811 (55%)]	Loss: 0.875333
Train Epoch: 33 [2048/2811 (73%)]	Loss: 0.841430
Train Epoch: 33 [2560/2811 (91%)]	Loss: 0.781252
val set: Average loss: 1.0153, Accuracy: 414/704 (58.81%)

Train Epoch: 34 [0/2811 (0%)]	Loss: 0.818197
Train Epoch: 34 [512/2811 (18%)]	Loss: 0.709423
Train Epoch: 34 [1024/2811 (36%)]	Loss: 0.726959
Train Epoch: 34 [1536/2811 (55%)]	Loss: 0.854349
Train Epoch: 34 [2048/2811 (73%)]	Loss: 0.949236
Train Epoch: 34 [2560/2811 (91%)]	Loss: 0.603975
val set: Average loss: 1.0862, Accuracy: 434/704 (61.65%)

Train Epoch: 35 [0/2811 (0%)]	Loss: 0.643740
Train Epoch: 35 [512/2811 (18%)]	Loss: 0.841662
Train Epoch: 35 [1024/2811 (36%)]	Loss: 0.818444
Train Epoch: 35 [1536/2811 (55%)]	Loss: 0.714916
Train Epoch: 35 [2048/2811 (73%)]	Loss: 0.880877
Train Epoch: 35 [2560/2811 (91%)]	Loss: 0.852917
val set: Average loss: 0.6602, Accuracy: 539/704 (76.56%)

Train Epoch: 36 [0/2811 (0%)]	Loss: 0.545482
Train Epoch: 36 [512/2811 (18%)]	Loss: 0.852692
Train Epoch: 36 [1024/2811 (36%)]	Loss: 0.741269
Train Epoch: 36 [1536/2811 (55%)]	Loss: 0.681979
Train Epoch: 36 [2048/2811 (73%)]	Loss: 0.557984
Train Epoch: 36 [2560/2811 (91%)]	Loss: 0.892148
val set: Average loss: 0.7497, Accuracy: 514/704 (73.01%)

Train Epoch: 37 [0/2811 (0%)]	Loss: 0.699462
Train Epoch: 37 [512/2811 (18%)]	Loss: 0.882952
Train Epoch: 37 [1024/2811 (36%)]	Loss: 0.711790
Train Epoch: 37 [1536/2811 (55%)]	Loss: 0.866510
Train Epoch: 37 [2048/2811 (73%)]	Loss: 0.643165
Train Epoch: 37 [2560/2811 (91%)]	Loss: 0.807665
val set: Average loss: 0.6786, Accuracy: 551/704 (78.27%)

Train Epoch: 38 [0/2811 (0%)]	Loss: 0.608815
Train Epoch: 38 [512/2811 (18%)]	Loss: 0.770223
Train Epoch: 38 [1024/2811 (36%)]	Loss: 0.700845
Train Epoch: 38 [1536/2811 (55%)]	Loss: 1.175622
Train Epoch: 38 [2048/2811 (73%)]	Loss: 0.818436
Train Epoch: 38 [2560/2811 (91%)]	Loss: 0.781179
val set: Average loss: 0.7101, Accuracy: 531/704 (75.43%)

Train Epoch: 39 [0/2811 (0%)]	Loss: 0.816681
Train Epoch: 39 [512/2811 (18%)]	Loss: 0.758668
Train Epoch: 39 [1024/2811 (36%)]	Loss: 0.670034
Train Epoch: 39 [1536/2811 (55%)]	Loss: 0.764461
Train Epoch: 39 [2048/2811 (73%)]	Loss: 0.665374
Train Epoch: 39 [2560/2811 (91%)]	Loss: 0.775801
val set: Average loss: 0.8113, Accuracy: 501/704 (71.16%)

Train Epoch: 40 [0/2811 (0%)]	Loss: 0.817223
Train Epoch: 40 [512/2811 (18%)]	Loss: 0.714475
Train Epoch: 40 [1024/2811 (36%)]	Loss: 0.743665
Train Epoch: 40 [1536/2811 (55%)]	Loss: 0.785111
Train Epoch: 40 [2048/2811 (73%)]	Loss: 0.614997
Train Epoch: 40 [2560/2811 (91%)]	Loss: 0.699038
val set: Average loss: 0.6519, Accuracy: 544/704 (77.27%)

Train Epoch: 41 [0/2811 (0%)]	Loss: 0.748128
Train Epoch: 41 [512/2811 (18%)]	Loss: 0.743023
Train Epoch: 41 [1024/2811 (36%)]	Loss: 0.662129
Train Epoch: 41 [1536/2811 (55%)]	Loss: 0.679916
Train Epoch: 41 [2048/2811 (73%)]	Loss: 0.566990
Train Epoch: 41 [2560/2811 (91%)]	Loss: 0.679266
val set: Average loss: 0.6174, Accuracy: 555/704 (78.84%)

Train Epoch: 42 [0/2811 (0%)]	Loss: 0.910663
Train Epoch: 42 [512/2811 (18%)]	Loss: 0.642704
Train Epoch: 42 [1024/2811 (36%)]	Loss: 0.606888
Train Epoch: 42 [1536/2811 (55%)]	Loss: 0.776218
Train Epoch: 42 [2048/2811 (73%)]	Loss: 0.796546
Train Epoch: 42 [2560/2811 (91%)]	Loss: 0.762481
val set: Average loss: 0.8423, Accuracy: 453/704 (64.35%)

Train Epoch: 43 [0/2811 (0%)]	Loss: 0.774732
Train Epoch: 43 [512/2811 (18%)]	Loss: 0.671517
Train Epoch: 43 [1024/2811 (36%)]	Loss: 0.955776
Train Epoch: 43 [1536/2811 (55%)]	Loss: 0.723714
Train Epoch: 43 [2048/2811 (73%)]	Loss: 0.612923
Train Epoch: 43 [2560/2811 (91%)]	Loss: 0.571453
val set: Average loss: 0.8973, Accuracy: 476/704 (67.61%)

Train Epoch: 44 [0/2811 (0%)]	Loss: 0.831987
Train Epoch: 44 [512/2811 (18%)]	Loss: 0.827096
Train Epoch: 44 [1024/2811 (36%)]	Loss: 0.788514
Train Epoch: 44 [1536/2811 (55%)]	Loss: 0.677893
Train Epoch: 44 [2048/2811 (73%)]	Loss: 0.713372
Train Epoch: 44 [2560/2811 (91%)]	Loss: 0.807915
val set: Average loss: 0.8291, Accuracy: 481/704 (68.32%)

Train Epoch: 45 [0/2811 (0%)]	Loss: 0.860796
Train Epoch: 45 [512/2811 (18%)]	Loss: 0.716864
Train Epoch: 45 [1024/2811 (36%)]	Loss: 0.968774
Train Epoch: 45 [1536/2811 (55%)]	Loss: 0.886019
Train Epoch: 45 [2048/2811 (73%)]	Loss: 0.849509
Train Epoch: 45 [2560/2811 (91%)]	Loss: 0.707767
val set: Average loss: 0.8460, Accuracy: 481/704 (68.32%)

Train Epoch: 46 [0/2811 (0%)]	Loss: 0.758631
Train Epoch: 46 [512/2811 (18%)]	Loss: 0.670735
Train Epoch: 46 [1024/2811 (36%)]	Loss: 0.976353
Train Epoch: 46 [1536/2811 (55%)]	Loss: 0.879906
Train Epoch: 46 [2048/2811 (73%)]	Loss: 0.839281
Train Epoch: 46 [2560/2811 (91%)]	Loss: 0.822902
val set: Average loss: 0.8930, Accuracy: 479/704 (68.04%)

Train Epoch: 47 [0/2811 (0%)]	Loss: 0.634493
Train Epoch: 47 [512/2811 (18%)]	Loss: 0.712842
Train Epoch: 47 [1024/2811 (36%)]	Loss: 0.676108
Train Epoch: 47 [1536/2811 (55%)]	Loss: 0.808655
Train Epoch: 47 [2048/2811 (73%)]	Loss: 0.840991
Train Epoch: 47 [2560/2811 (91%)]	Loss: 0.664775
val set: Average loss: 0.8546, Accuracy: 498/704 (70.74%)

Train Epoch: 48 [0/2811 (0%)]	Loss: 0.753210
Train Epoch: 48 [512/2811 (18%)]	Loss: 0.655755
Train Epoch: 48 [1024/2811 (36%)]	Loss: 0.494267
Train Epoch: 48 [1536/2811 (55%)]	Loss: 0.624551
Train Epoch: 48 [2048/2811 (73%)]	Loss: 0.626064
Train Epoch: 48 [2560/2811 (91%)]	Loss: 0.601110
val set: Average loss: 0.8532, Accuracy: 502/704 (71.31%)

Train Epoch: 49 [0/2811 (0%)]	Loss: 0.814962
Train Epoch: 49 [512/2811 (18%)]	Loss: 0.744647
Train Epoch: 49 [1024/2811 (36%)]	Loss: 0.630463
Train Epoch: 49 [1536/2811 (55%)]	Loss: 0.844658
Train Epoch: 49 [2048/2811 (73%)]	Loss: 0.631530
Train Epoch: 49 [2560/2811 (91%)]	Loss: 0.627567
val set: Average loss: 0.6411, Accuracy: 558/704 (79.26%)

Train Epoch: 50 [0/2811 (0%)]	Loss: 0.675711
Train Epoch: 50 [512/2811 (18%)]	Loss: 0.647200
Train Epoch: 50 [1024/2811 (36%)]	Loss: 0.882656
Train Epoch: 50 [1536/2811 (55%)]	Loss: 0.748918
Train Epoch: 50 [2048/2811 (73%)]	Loss: 0.824959
Train Epoch: 50 [2560/2811 (91%)]	Loss: 0.673118
val set: Average loss: 0.5980, Accuracy: 577/704 (81.96%)

Train Epoch: 51 [0/2811 (0%)]	Loss: 0.689386
Train Epoch: 51 [512/2811 (18%)]	Loss: 0.445036
Train Epoch: 51 [1024/2811 (36%)]	Loss: 0.627190
Train Epoch: 51 [1536/2811 (55%)]	Loss: 0.659882
Train Epoch: 51 [2048/2811 (73%)]	Loss: 0.718689
Train Epoch: 51 [2560/2811 (91%)]	Loss: 0.957390
val set: Average loss: 0.6587, Accuracy: 539/704 (76.56%)

Train Epoch: 52 [0/2811 (0%)]	Loss: 0.774084
Train Epoch: 52 [512/2811 (18%)]	Loss: 0.740277
Train Epoch: 52 [1024/2811 (36%)]	Loss: 0.618158
Train Epoch: 52 [1536/2811 (55%)]	Loss: 0.918906
Train Epoch: 52 [2048/2811 (73%)]	Loss: 0.852897
Train Epoch: 52 [2560/2811 (91%)]	Loss: 0.659133
val set: Average loss: 0.8685, Accuracy: 485/704 (68.89%)

Train Epoch: 53 [0/2811 (0%)]	Loss: 0.603370
Train Epoch: 53 [512/2811 (18%)]	Loss: 0.665041
Train Epoch: 53 [1024/2811 (36%)]	Loss: 1.013184
Train Epoch: 53 [1536/2811 (55%)]	Loss: 0.754819
Train Epoch: 53 [2048/2811 (73%)]	Loss: 0.705743
Train Epoch: 53 [2560/2811 (91%)]	Loss: 0.417137
val set: Average loss: 0.8236, Accuracy: 509/704 (72.30%)

Train Epoch: 54 [0/2811 (0%)]	Loss: 0.574621
Train Epoch: 54 [512/2811 (18%)]	Loss: 0.682821
Train Epoch: 54 [1024/2811 (36%)]	Loss: 0.686838
Train Epoch: 54 [1536/2811 (55%)]	Loss: 0.701078
Train Epoch: 54 [2048/2811 (73%)]	Loss: 0.797921
Train Epoch: 54 [2560/2811 (91%)]	Loss: 0.698030
val set: Average loss: 0.7369, Accuracy: 508/704 (72.16%)

Train Epoch: 55 [0/2811 (0%)]	Loss: 0.510849
Train Epoch: 55 [512/2811 (18%)]	Loss: 0.760039
Train Epoch: 55 [1024/2811 (36%)]	Loss: 0.845376
Train Epoch: 55 [1536/2811 (55%)]	Loss: 0.888983
Train Epoch: 55 [2048/2811 (73%)]	Loss: 0.978681
Train Epoch: 55 [2560/2811 (91%)]	Loss: 0.683099
val set: Average loss: 0.6666, Accuracy: 550/704 (78.12%)

Train Epoch: 56 [0/2811 (0%)]	Loss: 0.681451
Train Epoch: 56 [512/2811 (18%)]	Loss: 0.683922
Train Epoch: 56 [1024/2811 (36%)]	Loss: 0.802812
Train Epoch: 56 [1536/2811 (55%)]	Loss: 0.628761
Train Epoch: 56 [2048/2811 (73%)]	Loss: 0.782677
Train Epoch: 56 [2560/2811 (91%)]	Loss: 0.854421
val set: Average loss: 0.8748, Accuracy: 505/704 (71.73%)

Train Epoch: 57 [0/2811 (0%)]	Loss: 0.778013
Train Epoch: 57 [512/2811 (18%)]	Loss: 0.530944
Train Epoch: 57 [1024/2811 (36%)]	Loss: 0.641362
Train Epoch: 57 [1536/2811 (55%)]	Loss: 0.649272
Train Epoch: 57 [2048/2811 (73%)]	Loss: 0.620269
Train Epoch: 57 [2560/2811 (91%)]	Loss: 0.801825
val set: Average loss: 0.5944, Accuracy: 569/704 (80.82%)

Train Epoch: 58 [0/2811 (0%)]	Loss: 0.615567
Train Epoch: 58 [512/2811 (18%)]	Loss: 0.668607
Train Epoch: 58 [1024/2811 (36%)]	Loss: 0.532341
Train Epoch: 58 [1536/2811 (55%)]	Loss: 0.867308
Train Epoch: 58 [2048/2811 (73%)]	Loss: 0.734369
Train Epoch: 58 [2560/2811 (91%)]	Loss: 0.690496
val set: Average loss: 0.7092, Accuracy: 516/704 (73.30%)

Train Epoch: 59 [0/2811 (0%)]	Loss: 0.820032
Train Epoch: 59 [512/2811 (18%)]	Loss: 0.690745
Train Epoch: 59 [1024/2811 (36%)]	Loss: 0.852573
Train Epoch: 59 [1536/2811 (55%)]	Loss: 0.824640
Train Epoch: 59 [2048/2811 (73%)]	Loss: 0.770772
Train Epoch: 59 [2560/2811 (91%)]	Loss: 0.785838
val set: Average loss: 0.9374, Accuracy: 430/704 (61.08%)

Train Epoch: 60 [0/2811 (0%)]	Loss: 0.700168
Train Epoch: 60 [512/2811 (18%)]	Loss: 0.495212
Train Epoch: 60 [1024/2811 (36%)]	Loss: 0.648017
Train Epoch: 60 [1536/2811 (55%)]	Loss: 0.655263
Train Epoch: 60 [2048/2811 (73%)]	Loss: 0.685133
Train Epoch: 60 [2560/2811 (91%)]	Loss: 0.487630
val set: Average loss: 0.6331, Accuracy: 552/704 (78.41%)

Train Epoch: 61 [0/2811 (0%)]	Loss: 0.627170
Train Epoch: 61 [512/2811 (18%)]	Loss: 0.767268
Train Epoch: 61 [1024/2811 (36%)]	Loss: 0.618836
Train Epoch: 61 [1536/2811 (55%)]	Loss: 0.659025
Train Epoch: 61 [2048/2811 (73%)]	Loss: 0.573114
Train Epoch: 61 [2560/2811 (91%)]	Loss: 0.829723
val set: Average loss: 0.7117, Accuracy: 537/704 (76.28%)

Train Epoch: 62 [0/2811 (0%)]	Loss: 0.738514
Train Epoch: 62 [512/2811 (18%)]	Loss: 0.591791
Train Epoch: 62 [1024/2811 (36%)]	Loss: 0.676555
Train Epoch: 62 [1536/2811 (55%)]	Loss: 0.734182
Train Epoch: 62 [2048/2811 (73%)]	Loss: 0.702335
Train Epoch: 62 [2560/2811 (91%)]	Loss: 0.625642
val set: Average loss: 0.7915, Accuracy: 496/704 (70.45%)

Train Epoch: 63 [0/2811 (0%)]	Loss: 0.630390
Train Epoch: 63 [512/2811 (18%)]	Loss: 0.769121
Train Epoch: 63 [1024/2811 (36%)]	Loss: 0.634342
Train Epoch: 63 [1536/2811 (55%)]	Loss: 0.643496
Train Epoch: 63 [2048/2811 (73%)]	Loss: 0.612579
Train Epoch: 63 [2560/2811 (91%)]	Loss: 0.630933
val set: Average loss: 0.6881, Accuracy: 539/704 (76.56%)

Train Epoch: 64 [0/2811 (0%)]	Loss: 0.548199
Train Epoch: 64 [512/2811 (18%)]	Loss: 1.220413
Train Epoch: 64 [1024/2811 (36%)]	Loss: 0.661299
Train Epoch: 64 [1536/2811 (55%)]	Loss: 0.465877
Train Epoch: 64 [2048/2811 (73%)]	Loss: 0.530598
Train Epoch: 64 [2560/2811 (91%)]	Loss: 0.831501
val set: Average loss: 0.7757, Accuracy: 507/704 (72.02%)

Train Epoch: 65 [0/2811 (0%)]	Loss: 0.508900
Train Epoch: 65 [512/2811 (18%)]	Loss: 0.696233
Train Epoch: 65 [1024/2811 (36%)]	Loss: 0.710013
Train Epoch: 65 [1536/2811 (55%)]	Loss: 0.538573
Train Epoch: 65 [2048/2811 (73%)]	Loss: 0.638123
Train Epoch: 65 [2560/2811 (91%)]	Loss: 0.640594
val set: Average loss: 0.9371, Accuracy: 484/704 (68.75%)

Train Epoch: 66 [0/2811 (0%)]	Loss: 0.663303
Train Epoch: 66 [512/2811 (18%)]	Loss: 0.683078
Train Epoch: 66 [1024/2811 (36%)]	Loss: 0.608738
Train Epoch: 66 [1536/2811 (55%)]	Loss: 0.545494
Train Epoch: 66 [2048/2811 (73%)]	Loss: 0.560292
Train Epoch: 66 [2560/2811 (91%)]	Loss: 0.641335
val set: Average loss: 0.6848, Accuracy: 539/704 (76.56%)

Train Epoch: 67 [0/2811 (0%)]	Loss: 0.704700
Train Epoch: 67 [512/2811 (18%)]	Loss: 0.629561
Train Epoch: 67 [1024/2811 (36%)]	Loss: 0.431505
Train Epoch: 67 [1536/2811 (55%)]	Loss: 0.620383
Train Epoch: 67 [2048/2811 (73%)]	Loss: 0.626984
Train Epoch: 67 [2560/2811 (91%)]	Loss: 0.530483
val set: Average loss: 0.9974, Accuracy: 453/704 (64.35%)

Train Epoch: 68 [0/2811 (0%)]	Loss: 0.556561
Train Epoch: 68 [512/2811 (18%)]	Loss: 0.700515
Train Epoch: 68 [1024/2811 (36%)]	Loss: 0.696973
Train Epoch: 68 [1536/2811 (55%)]	Loss: 0.544122
Train Epoch: 68 [2048/2811 (73%)]	Loss: 0.400488
Train Epoch: 68 [2560/2811 (91%)]	Loss: 0.550383
val set: Average loss: 0.8580, Accuracy: 504/704 (71.59%)

Train Epoch: 69 [0/2811 (0%)]	Loss: 0.461838
Train Epoch: 69 [512/2811 (18%)]	Loss: 0.543203
Train Epoch: 69 [1024/2811 (36%)]	Loss: 0.526387
Train Epoch: 69 [1536/2811 (55%)]	Loss: 0.913584
Train Epoch: 69 [2048/2811 (73%)]	Loss: 0.788571
Train Epoch: 69 [2560/2811 (91%)]	Loss: 0.556997
val set: Average loss: 0.7151, Accuracy: 530/704 (75.28%)

Train Epoch: 70 [0/2811 (0%)]	Loss: 0.598285
Train Epoch: 70 [512/2811 (18%)]	Loss: 0.487519
Train Epoch: 70 [1024/2811 (36%)]	Loss: 0.854664
Train Epoch: 70 [1536/2811 (55%)]	Loss: 0.670795
Train Epoch: 70 [2048/2811 (73%)]	Loss: 0.418451
Train Epoch: 70 [2560/2811 (91%)]	Loss: 0.438460
val set: Average loss: 0.8383, Accuracy: 492/704 (69.89%)

Train Epoch: 71 [0/2811 (0%)]	Loss: 0.714666
Train Epoch: 71 [512/2811 (18%)]	Loss: 0.460162
Train Epoch: 71 [1024/2811 (36%)]	Loss: 0.822775
Train Epoch: 71 [1536/2811 (55%)]	Loss: 0.786383
Train Epoch: 71 [2048/2811 (73%)]	Loss: 0.503524
Train Epoch: 71 [2560/2811 (91%)]	Loss: 0.736541
val set: Average loss: 0.7599, Accuracy: 515/704 (73.15%)

Train Epoch: 72 [0/2811 (0%)]	Loss: 0.818473
Train Epoch: 72 [512/2811 (18%)]	Loss: 0.607594
Train Epoch: 72 [1024/2811 (36%)]	Loss: 0.653295
Train Epoch: 72 [1536/2811 (55%)]	Loss: 0.503926
Train Epoch: 72 [2048/2811 (73%)]	Loss: 0.441520
Train Epoch: 72 [2560/2811 (91%)]	Loss: 0.471133
val set: Average loss: 0.6484, Accuracy: 546/704 (77.56%)

Train Epoch: 73 [0/2811 (0%)]	Loss: 0.746359
Train Epoch: 73 [512/2811 (18%)]	Loss: 0.561166
Train Epoch: 73 [1024/2811 (36%)]	Loss: 0.516853
Train Epoch: 73 [1536/2811 (55%)]	Loss: 0.555914
Train Epoch: 73 [2048/2811 (73%)]	Loss: 0.463113
Train Epoch: 73 [2560/2811 (91%)]	Loss: 0.661176
val set: Average loss: 0.6837, Accuracy: 535/704 (75.99%)

Train Epoch: 74 [0/2811 (0%)]	Loss: 0.854178
Train Epoch: 74 [512/2811 (18%)]	Loss: 0.560940
Train Epoch: 74 [1024/2811 (36%)]	Loss: 0.906971
Train Epoch: 74 [1536/2811 (55%)]	Loss: 0.863633
Train Epoch: 74 [2048/2811 (73%)]	Loss: 0.444773
Train Epoch: 74 [2560/2811 (91%)]	Loss: 0.498360
val set: Average loss: 0.6728, Accuracy: 539/704 (76.56%)

Train Epoch: 75 [0/2811 (0%)]	Loss: 0.806681
Train Epoch: 75 [512/2811 (18%)]	Loss: 0.731302
Train Epoch: 75 [1024/2811 (36%)]	Loss: 0.696826
Train Epoch: 75 [1536/2811 (55%)]	Loss: 0.563548
Train Epoch: 75 [2048/2811 (73%)]	Loss: 0.409122
Train Epoch: 75 [2560/2811 (91%)]	Loss: 0.613375
val set: Average loss: 0.6802, Accuracy: 535/704 (75.99%)

Train Epoch: 76 [0/2811 (0%)]	Loss: 0.360054
Train Epoch: 76 [512/2811 (18%)]	Loss: 0.700774
Train Epoch: 76 [1024/2811 (36%)]	Loss: 0.522025
Train Epoch: 76 [1536/2811 (55%)]	Loss: 0.588121
Train Epoch: 76 [2048/2811 (73%)]	Loss: 0.577000
Train Epoch: 76 [2560/2811 (91%)]	Loss: 0.638088
val set: Average loss: 0.8111, Accuracy: 502/704 (71.31%)

Train Epoch: 77 [0/2811 (0%)]	Loss: 0.335187
Train Epoch: 77 [512/2811 (18%)]	Loss: 0.513677
Train Epoch: 77 [1024/2811 (36%)]	Loss: 0.480850
Train Epoch: 77 [1536/2811 (55%)]	Loss: 0.685982
Train Epoch: 77 [2048/2811 (73%)]	Loss: 0.495007
Train Epoch: 77 [2560/2811 (91%)]	Loss: 0.772952
val set: Average loss: 0.8641, Accuracy: 474/704 (67.33%)

Train Epoch: 78 [0/2811 (0%)]	Loss: 0.793732
Train Epoch: 78 [512/2811 (18%)]	Loss: 0.562154
Train Epoch: 78 [1024/2811 (36%)]	Loss: 0.654601
Train Epoch: 78 [1536/2811 (55%)]	Loss: 0.702869
Train Epoch: 78 [2048/2811 (73%)]	Loss: 0.721963
Train Epoch: 78 [2560/2811 (91%)]	Loss: 0.616592
val set: Average loss: 0.6153, Accuracy: 564/704 (80.11%)

Train Epoch: 79 [0/2811 (0%)]	Loss: 0.344229
Train Epoch: 79 [512/2811 (18%)]	Loss: 0.474079
Train Epoch: 79 [1024/2811 (36%)]	Loss: 0.421708
Train Epoch: 79 [1536/2811 (55%)]	Loss: 0.613297
Train Epoch: 79 [2048/2811 (73%)]	Loss: 0.811836
Train Epoch: 79 [2560/2811 (91%)]	Loss: 0.556451
val set: Average loss: 0.7295, Accuracy: 525/704 (74.57%)

Train Epoch: 80 [0/2811 (0%)]	Loss: 0.363940
Train Epoch: 80 [512/2811 (18%)]	Loss: 0.643928
Train Epoch: 80 [1024/2811 (36%)]	Loss: 0.434519
Train Epoch: 80 [1536/2811 (55%)]	Loss: 0.638933
Train Epoch: 80 [2048/2811 (73%)]	Loss: 0.752214
Train Epoch: 80 [2560/2811 (91%)]	Loss: 0.682361
val set: Average loss: 0.6000, Accuracy: 564/704 (80.11%)

Train Epoch: 81 [0/2811 (0%)]	Loss: 0.746364
Train Epoch: 81 [512/2811 (18%)]	Loss: 0.353367
Train Epoch: 81 [1024/2811 (36%)]	Loss: 0.612789
Train Epoch: 81 [1536/2811 (55%)]	Loss: 0.564572
Train Epoch: 81 [2048/2811 (73%)]	Loss: 0.359242
Train Epoch: 81 [2560/2811 (91%)]	Loss: 0.570323
val set: Average loss: 0.5892, Accuracy: 570/704 (80.97%)

Train Epoch: 82 [0/2811 (0%)]	Loss: 0.484081
Train Epoch: 82 [512/2811 (18%)]	Loss: 0.643201
Train Epoch: 82 [1024/2811 (36%)]	Loss: 0.615519
Train Epoch: 82 [1536/2811 (55%)]	Loss: 0.401930
Train Epoch: 82 [2048/2811 (73%)]	Loss: 0.668718
Train Epoch: 82 [2560/2811 (91%)]	Loss: 0.597791
val set: Average loss: 0.6901, Accuracy: 532/704 (75.57%)

Train Epoch: 83 [0/2811 (0%)]	Loss: 0.720790
Train Epoch: 83 [512/2811 (18%)]	Loss: 0.647290
Train Epoch: 83 [1024/2811 (36%)]	Loss: 0.807556
Train Epoch: 83 [1536/2811 (55%)]	Loss: 0.549282
Train Epoch: 83 [2048/2811 (73%)]	Loss: 0.603131
Train Epoch: 83 [2560/2811 (91%)]	Loss: 0.745789
val set: Average loss: 0.7093, Accuracy: 526/704 (74.72%)

Train Epoch: 84 [0/2811 (0%)]	Loss: 0.571832
Train Epoch: 84 [512/2811 (18%)]	Loss: 0.500376
Train Epoch: 84 [1024/2811 (36%)]	Loss: 0.784452
Train Epoch: 84 [1536/2811 (55%)]	Loss: 0.761735
Train Epoch: 84 [2048/2811 (73%)]	Loss: 0.639962
Train Epoch: 84 [2560/2811 (91%)]	Loss: 0.490673
val set: Average loss: 0.7165, Accuracy: 521/704 (74.01%)

Train Epoch: 85 [0/2811 (0%)]	Loss: 0.533001
Train Epoch: 85 [512/2811 (18%)]	Loss: 0.553450
Train Epoch: 85 [1024/2811 (36%)]	Loss: 0.543192
Train Epoch: 85 [1536/2811 (55%)]	Loss: 0.593962
Train Epoch: 85 [2048/2811 (73%)]	Loss: 0.741629
Train Epoch: 85 [2560/2811 (91%)]	Loss: 0.506740
val set: Average loss: 0.7053, Accuracy: 537/704 (76.28%)

Train Epoch: 86 [0/2811 (0%)]	Loss: 0.753541
Train Epoch: 86 [512/2811 (18%)]	Loss: 0.590451
Train Epoch: 86 [1024/2811 (36%)]	Loss: 0.578196
Train Epoch: 86 [1536/2811 (55%)]	Loss: 0.500160
Train Epoch: 86 [2048/2811 (73%)]	Loss: 0.601787
Train Epoch: 86 [2560/2811 (91%)]	Loss: 0.576654
val set: Average loss: 0.7451, Accuracy: 518/704 (73.58%)

Train Epoch: 87 [0/2811 (0%)]	Loss: 0.783122
Train Epoch: 87 [512/2811 (18%)]	Loss: 0.462459
Train Epoch: 87 [1024/2811 (36%)]	Loss: 0.758935
Train Epoch: 87 [1536/2811 (55%)]	Loss: 0.536627
Train Epoch: 87 [2048/2811 (73%)]	Loss: 0.674985
Train Epoch: 87 [2560/2811 (91%)]	Loss: 0.570266
val set: Average loss: 0.6830, Accuracy: 537/704 (76.28%)

Train Epoch: 88 [0/2811 (0%)]	Loss: 0.408696
Train Epoch: 88 [512/2811 (18%)]	Loss: 0.700679
Train Epoch: 88 [1024/2811 (36%)]	Loss: 0.531575
Train Epoch: 88 [1536/2811 (55%)]	Loss: 0.558715
Train Epoch: 88 [2048/2811 (73%)]	Loss: 0.479716
Train Epoch: 88 [2560/2811 (91%)]	Loss: 0.514161
val set: Average loss: 0.6951, Accuracy: 525/704 (74.57%)

Train Epoch: 89 [0/2811 (0%)]	Loss: 0.553557
Train Epoch: 89 [512/2811 (18%)]	Loss: 0.706317
Train Epoch: 89 [1024/2811 (36%)]	Loss: 0.501075
Train Epoch: 89 [1536/2811 (55%)]	Loss: 0.504326
Train Epoch: 89 [2048/2811 (73%)]	Loss: 0.507434
Train Epoch: 89 [2560/2811 (91%)]	Loss: 0.721209
val set: Average loss: 0.7539, Accuracy: 534/704 (75.85%)

Train Epoch: 90 [0/2811 (0%)]	Loss: 0.511788
Train Epoch: 90 [512/2811 (18%)]	Loss: 0.607733
Train Epoch: 90 [1024/2811 (36%)]	Loss: 0.583795
Train Epoch: 90 [1536/2811 (55%)]	Loss: 0.379130
Train Epoch: 90 [2048/2811 (73%)]	Loss: 0.584103
Train Epoch: 90 [2560/2811 (91%)]	Loss: 0.431510
val set: Average loss: 0.7260, Accuracy: 528/704 (75.00%)

Train Epoch: 91 [0/2811 (0%)]	Loss: 0.553851
Train Epoch: 91 [512/2811 (18%)]	Loss: 0.659189
Train Epoch: 91 [1024/2811 (36%)]	Loss: 0.766671
Train Epoch: 91 [1536/2811 (55%)]	Loss: 0.515836
Train Epoch: 91 [2048/2811 (73%)]	Loss: 0.543227
Train Epoch: 91 [2560/2811 (91%)]	Loss: 0.664711
val set: Average loss: 0.9092, Accuracy: 494/704 (70.17%)

Train Epoch: 92 [0/2811 (0%)]	Loss: 0.461244
Train Epoch: 92 [512/2811 (18%)]	Loss: 0.540775
Train Epoch: 92 [1024/2811 (36%)]	Loss: 0.585623
Train Epoch: 92 [1536/2811 (55%)]	Loss: 0.561702
Train Epoch: 92 [2048/2811 (73%)]	Loss: 0.574182
Train Epoch: 92 [2560/2811 (91%)]	Loss: 0.618195
val set: Average loss: 0.7360, Accuracy: 527/704 (74.86%)

Train Epoch: 93 [0/2811 (0%)]	Loss: 0.567659
Train Epoch: 93 [512/2811 (18%)]	Loss: 0.613225
Train Epoch: 93 [1024/2811 (36%)]	Loss: 0.332810
Train Epoch: 93 [1536/2811 (55%)]	Loss: 0.465913
Train Epoch: 93 [2048/2811 (73%)]	Loss: 0.409372
Train Epoch: 93 [2560/2811 (91%)]	Loss: 0.483713
val set: Average loss: 0.8309, Accuracy: 486/704 (69.03%)

Train Epoch: 94 [0/2811 (0%)]	Loss: 0.544911
Train Epoch: 94 [512/2811 (18%)]	Loss: 0.530360
Train Epoch: 94 [1024/2811 (36%)]	Loss: 0.460184
Train Epoch: 94 [1536/2811 (55%)]	Loss: 0.313926
Train Epoch: 94 [2048/2811 (73%)]	Loss: 0.552451
Train Epoch: 94 [2560/2811 (91%)]	Loss: 0.493559
val set: Average loss: 0.6602, Accuracy: 552/704 (78.41%)

Train Epoch: 95 [0/2811 (0%)]	Loss: 0.399973
Train Epoch: 95 [512/2811 (18%)]	Loss: 0.326263
Train Epoch: 95 [1024/2811 (36%)]	Loss: 0.655100
Train Epoch: 95 [1536/2811 (55%)]	Loss: 0.486181
Train Epoch: 95 [2048/2811 (73%)]	Loss: 0.489062
Train Epoch: 95 [2560/2811 (91%)]	Loss: 0.579393
val set: Average loss: 0.7585, Accuracy: 519/704 (73.72%)

Train Epoch: 96 [0/2811 (0%)]	Loss: 0.719878
Train Epoch: 96 [512/2811 (18%)]	Loss: 0.543959
Train Epoch: 96 [1024/2811 (36%)]	Loss: 0.611767
Train Epoch: 96 [1536/2811 (55%)]	Loss: 0.603598
Train Epoch: 96 [2048/2811 (73%)]	Loss: 0.552238
Train Epoch: 96 [2560/2811 (91%)]	Loss: 0.584959
val set: Average loss: 0.6294, Accuracy: 541/704 (76.85%)

Train Epoch: 97 [0/2811 (0%)]	Loss: 0.696945
Train Epoch: 97 [512/2811 (18%)]	Loss: 0.548856
Train Epoch: 97 [1024/2811 (36%)]	Loss: 0.672660
Train Epoch: 97 [1536/2811 (55%)]	Loss: 0.509762
Train Epoch: 97 [2048/2811 (73%)]	Loss: 0.503718
Train Epoch: 97 [2560/2811 (91%)]	Loss: 0.544841
val set: Average loss: 0.6908, Accuracy: 532/704 (75.57%)

Train Epoch: 98 [0/2811 (0%)]	Loss: 0.553408
Train Epoch: 98 [512/2811 (18%)]	Loss: 0.566776
Train Epoch: 98 [1024/2811 (36%)]	Loss: 0.988154
Train Epoch: 98 [1536/2811 (55%)]	Loss: 0.732962
Train Epoch: 98 [2048/2811 (73%)]	Loss: 0.696370
Train Epoch: 98 [2560/2811 (91%)]	Loss: 0.622244
val set: Average loss: 0.6876, Accuracy: 534/704 (75.85%)

Train Epoch: 99 [0/2811 (0%)]	Loss: 0.615557
Train Epoch: 99 [512/2811 (18%)]	Loss: 0.468563
Train Epoch: 99 [1024/2811 (36%)]	Loss: 0.483889
Train Epoch: 99 [1536/2811 (55%)]	Loss: 0.447409
Train Epoch: 99 [2048/2811 (73%)]	Loss: 0.504063
Train Epoch: 99 [2560/2811 (91%)]	Loss: 0.362015
val set: Average loss: 0.8209, Accuracy: 492/704 (69.89%)

Train Epoch: 100 [0/2811 (0%)]	Loss: 0.694196
Train Epoch: 100 [512/2811 (18%)]	Loss: 0.580925
Train Epoch: 100 [1024/2811 (36%)]	Loss: 0.728729
Train Epoch: 100 [1536/2811 (55%)]	Loss: 0.680517
Train Epoch: 100 [2048/2811 (73%)]	Loss: 0.497039
Train Epoch: 100 [2560/2811 (91%)]	Loss: 0.353085
val set: Average loss: 0.6982, Accuracy: 547/704 (77.70%)

Train Epoch: 101 [0/2811 (0%)]	Loss: 0.727758
Train Epoch: 101 [512/2811 (18%)]	Loss: 0.614268
Train Epoch: 101 [1024/2811 (36%)]	Loss: 0.480696
Train Epoch: 101 [1536/2811 (55%)]	Loss: 0.416414
Train Epoch: 101 [2048/2811 (73%)]	Loss: 0.623588
Train Epoch: 101 [2560/2811 (91%)]	Loss: 0.520797
val set: Average loss: 0.7051, Accuracy: 531/704 (75.43%)

Train Epoch: 102 [0/2811 (0%)]	Loss: 0.566927
Train Epoch: 102 [512/2811 (18%)]	Loss: 0.885186
Train Epoch: 102 [1024/2811 (36%)]	Loss: 0.536822
Train Epoch: 102 [1536/2811 (55%)]	Loss: 0.474602
Train Epoch: 102 [2048/2811 (73%)]	Loss: 0.444066
Train Epoch: 102 [2560/2811 (91%)]	Loss: 0.379921
val set: Average loss: 0.6155, Accuracy: 567/704 (80.54%)

Train Epoch: 103 [0/2811 (0%)]	Loss: 0.526629
Train Epoch: 103 [512/2811 (18%)]	Loss: 0.492258
Train Epoch: 103 [1024/2811 (36%)]	Loss: 0.430107
Train Epoch: 103 [1536/2811 (55%)]	Loss: 0.646216
Train Epoch: 103 [2048/2811 (73%)]	Loss: 0.466196
Train Epoch: 103 [2560/2811 (91%)]	Loss: 0.553174
val set: Average loss: 0.7456, Accuracy: 538/704 (76.42%)

Train Epoch: 104 [0/2811 (0%)]	Loss: 0.730947
Train Epoch: 104 [512/2811 (18%)]	Loss: 0.408497
Train Epoch: 104 [1024/2811 (36%)]	Loss: 0.399106
Train Epoch: 104 [1536/2811 (55%)]	Loss: 0.763947
Train Epoch: 104 [2048/2811 (73%)]	Loss: 0.557056
Train Epoch: 104 [2560/2811 (91%)]	Loss: 0.628510
val set: Average loss: 0.6367, Accuracy: 548/704 (77.84%)

Train Epoch: 105 [0/2811 (0%)]	Loss: 0.529231
Train Epoch: 105 [512/2811 (18%)]	Loss: 0.723957
Train Epoch: 105 [1024/2811 (36%)]	Loss: 0.580415
Train Epoch: 105 [1536/2811 (55%)]	Loss: 0.882501
Train Epoch: 105 [2048/2811 (73%)]	Loss: 0.484162
Train Epoch: 105 [2560/2811 (91%)]	Loss: 0.503112
val set: Average loss: 0.6146, Accuracy: 553/704 (78.55%)

Train Epoch: 106 [0/2811 (0%)]	Loss: 0.520574
Train Epoch: 106 [512/2811 (18%)]	Loss: 0.638557
Train Epoch: 106 [1024/2811 (36%)]	Loss: 0.318502
Train Epoch: 106 [1536/2811 (55%)]	Loss: 0.460333
Train Epoch: 106 [2048/2811 (73%)]	Loss: 0.711813
Train Epoch: 106 [2560/2811 (91%)]	Loss: 0.704356
val set: Average loss: 0.7138, Accuracy: 521/704 (74.01%)

Train Epoch: 107 [0/2811 (0%)]	Loss: 0.630844
Train Epoch: 107 [512/2811 (18%)]	Loss: 0.703754
Train Epoch: 107 [1024/2811 (36%)]	Loss: 0.364837
Train Epoch: 107 [1536/2811 (55%)]	Loss: 0.498242
Train Epoch: 107 [2048/2811 (73%)]	Loss: 0.753241
Train Epoch: 107 [2560/2811 (91%)]	Loss: 0.677300
val set: Average loss: 0.8363, Accuracy: 484/704 (68.75%)

Train Epoch: 108 [0/2811 (0%)]	Loss: 0.526034
Train Epoch: 108 [512/2811 (18%)]	Loss: 0.463525
Train Epoch: 108 [1024/2811 (36%)]	Loss: 0.683956
Train Epoch: 108 [1536/2811 (55%)]	Loss: 0.591004
Train Epoch: 108 [2048/2811 (73%)]	Loss: 0.723991
Train Epoch: 108 [2560/2811 (91%)]	Loss: 0.829831
val set: Average loss: 0.6417, Accuracy: 550/704 (78.12%)

Train Epoch: 109 [0/2811 (0%)]	Loss: 0.455011
Train Epoch: 109 [512/2811 (18%)]	Loss: 0.508570
Train Epoch: 109 [1024/2811 (36%)]	Loss: 0.550389
Train Epoch: 109 [1536/2811 (55%)]	Loss: 0.512040
Train Epoch: 109 [2048/2811 (73%)]	Loss: 0.443450
Train Epoch: 109 [2560/2811 (91%)]	Loss: 0.366424
val set: Average loss: 0.6432, Accuracy: 552/704 (78.41%)

Train Epoch: 110 [0/2811 (0%)]	Loss: 0.570761
Train Epoch: 110 [512/2811 (18%)]	Loss: 0.364475
Train Epoch: 110 [1024/2811 (36%)]	Loss: 0.497846
Train Epoch: 110 [1536/2811 (55%)]	Loss: 0.465126
Train Epoch: 110 [2048/2811 (73%)]	Loss: 0.424240
Train Epoch: 110 [2560/2811 (91%)]	Loss: 0.524121
val set: Average loss: 0.6667, Accuracy: 536/704 (76.14%)

Train Epoch: 111 [0/2811 (0%)]	Loss: 0.493316
Train Epoch: 111 [512/2811 (18%)]	Loss: 0.569956
Train Epoch: 111 [1024/2811 (36%)]	Loss: 0.791819
Train Epoch: 111 [1536/2811 (55%)]	Loss: 0.431089
Train Epoch: 111 [2048/2811 (73%)]	Loss: 0.574126
Train Epoch: 111 [2560/2811 (91%)]	Loss: 0.364465
val set: Average loss: 0.7255, Accuracy: 520/704 (73.86%)

Train Epoch: 112 [0/2811 (0%)]	Loss: 0.786951
Train Epoch: 112 [512/2811 (18%)]	Loss: 0.624252
Train Epoch: 112 [1024/2811 (36%)]	Loss: 0.344999
Train Epoch: 112 [1536/2811 (55%)]	Loss: 0.411484
Train Epoch: 112 [2048/2811 (73%)]	Loss: 0.880122
Train Epoch: 112 [2560/2811 (91%)]	Loss: 0.604694
val set: Average loss: 0.6203, Accuracy: 549/704 (77.98%)

Train Epoch: 113 [0/2811 (0%)]	Loss: 0.433088
Train Epoch: 113 [512/2811 (18%)]	Loss: 0.480976
Train Epoch: 113 [1024/2811 (36%)]	Loss: 0.654781
Train Epoch: 113 [1536/2811 (55%)]	Loss: 0.466242
Train Epoch: 113 [2048/2811 (73%)]	Loss: 0.342487
Train Epoch: 113 [2560/2811 (91%)]	Loss: 0.520505
val set: Average loss: 0.7122, Accuracy: 533/704 (75.71%)

Train Epoch: 114 [0/2811 (0%)]	Loss: 0.418648
Train Epoch: 114 [512/2811 (18%)]	Loss: 0.596200
Train Epoch: 114 [1024/2811 (36%)]	Loss: 0.648851
Train Epoch: 114 [1536/2811 (55%)]	Loss: 0.451436
Train Epoch: 114 [2048/2811 (73%)]	Loss: 0.393003
Train Epoch: 114 [2560/2811 (91%)]	Loss: 0.440015
val set: Average loss: 0.7374, Accuracy: 522/704 (74.15%)

Train Epoch: 115 [0/2811 (0%)]	Loss: 0.471363
Train Epoch: 115 [512/2811 (18%)]	Loss: 0.362839
Train Epoch: 115 [1024/2811 (36%)]	Loss: 0.700211
Train Epoch: 115 [1536/2811 (55%)]	Loss: 0.690881
Train Epoch: 115 [2048/2811 (73%)]	Loss: 0.458989
Train Epoch: 115 [2560/2811 (91%)]	Loss: 0.534311
val set: Average loss: 0.7279, Accuracy: 531/704 (75.43%)

Train Epoch: 116 [0/2811 (0%)]	Loss: 0.414555
Train Epoch: 116 [512/2811 (18%)]	Loss: 0.410657
Train Epoch: 116 [1024/2811 (36%)]	Loss: 0.398476
Train Epoch: 116 [1536/2811 (55%)]	Loss: 0.596952
Train Epoch: 116 [2048/2811 (73%)]	Loss: 0.531312
Train Epoch: 116 [2560/2811 (91%)]	Loss: 0.707632
val set: Average loss: 0.6463, Accuracy: 550/704 (78.12%)

Train Epoch: 117 [0/2811 (0%)]	Loss: 0.385540
Train Epoch: 117 [512/2811 (18%)]	Loss: 0.412493
Train Epoch: 117 [1024/2811 (36%)]	Loss: 0.579049
Train Epoch: 117 [1536/2811 (55%)]	Loss: 0.301657
Train Epoch: 117 [2048/2811 (73%)]	Loss: 0.511941
Train Epoch: 117 [2560/2811 (91%)]	Loss: 0.472210
val set: Average loss: 0.7119, Accuracy: 526/704 (74.72%)

Train Epoch: 118 [0/2811 (0%)]	Loss: 0.490140
Train Epoch: 118 [512/2811 (18%)]	Loss: 0.466823
Train Epoch: 118 [1024/2811 (36%)]	Loss: 0.332807
Train Epoch: 118 [1536/2811 (55%)]	Loss: 0.344369
Train Epoch: 118 [2048/2811 (73%)]	Loss: 0.576371
Train Epoch: 118 [2560/2811 (91%)]	Loss: 0.548138
val set: Average loss: 0.7223, Accuracy: 532/704 (75.57%)

Train Epoch: 119 [0/2811 (0%)]	Loss: 0.605489
Train Epoch: 119 [512/2811 (18%)]	Loss: 0.486483
Train Epoch: 119 [1024/2811 (36%)]	Loss: 0.693818
Train Epoch: 119 [1536/2811 (55%)]	Loss: 0.569356
Train Epoch: 119 [2048/2811 (73%)]	Loss: 0.445125
Train Epoch: 119 [2560/2811 (91%)]	Loss: 0.616714
val set: Average loss: 0.6296, Accuracy: 554/704 (78.69%)

Train Epoch: 120 [0/2811 (0%)]	Loss: 0.465761
Train Epoch: 120 [512/2811 (18%)]	Loss: 0.450619
Train Epoch: 120 [1024/2811 (36%)]	Loss: 0.459935
Train Epoch: 120 [1536/2811 (55%)]	Loss: 0.574026
Train Epoch: 120 [2048/2811 (73%)]	Loss: 0.639412
Train Epoch: 120 [2560/2811 (91%)]	Loss: 0.507874
val set: Average loss: 0.6142, Accuracy: 560/704 (79.55%)

Train Epoch: 121 [0/2811 (0%)]	Loss: 0.438245
Train Epoch: 121 [512/2811 (18%)]	Loss: 0.381845
Train Epoch: 121 [1024/2811 (36%)]	Loss: 0.492366
Train Epoch: 121 [1536/2811 (55%)]	Loss: 0.433467
Train Epoch: 121 [2048/2811 (73%)]	Loss: 0.483231
Train Epoch: 121 [2560/2811 (91%)]	Loss: 0.542852
val set: Average loss: 0.5399, Accuracy: 576/704 (81.82%)

Train Epoch: 122 [0/2811 (0%)]	Loss: 0.391812
Train Epoch: 122 [512/2811 (18%)]	Loss: 0.507517
Train Epoch: 122 [1024/2811 (36%)]	Loss: 0.525097
Train Epoch: 122 [1536/2811 (55%)]	Loss: 0.523059
Train Epoch: 122 [2048/2811 (73%)]	Loss: 0.544794
Train Epoch: 122 [2560/2811 (91%)]	Loss: 0.542230
val set: Average loss: 0.6731, Accuracy: 539/704 (76.56%)

Train Epoch: 123 [0/2811 (0%)]	Loss: 0.613063
Train Epoch: 123 [512/2811 (18%)]	Loss: 0.430518
Train Epoch: 123 [1024/2811 (36%)]	Loss: 0.472615
Train Epoch: 123 [1536/2811 (55%)]	Loss: 0.755543
Train Epoch: 123 [2048/2811 (73%)]	Loss: 0.501620
Train Epoch: 123 [2560/2811 (91%)]	Loss: 0.490549
val set: Average loss: 0.5974, Accuracy: 567/704 (80.54%)

Train Epoch: 124 [0/2811 (0%)]	Loss: 0.403328
Train Epoch: 124 [512/2811 (18%)]	Loss: 0.399142
Train Epoch: 124 [1024/2811 (36%)]	Loss: 0.379907
Train Epoch: 124 [1536/2811 (55%)]	Loss: 0.429042
Train Epoch: 124 [2048/2811 (73%)]	Loss: 0.678048
Train Epoch: 124 [2560/2811 (91%)]	Loss: 0.504410
val set: Average loss: 0.6760, Accuracy: 519/704 (73.72%)

Train Epoch: 125 [0/2811 (0%)]	Loss: 0.477453
Train Epoch: 125 [512/2811 (18%)]	Loss: 0.631171
Train Epoch: 125 [1024/2811 (36%)]	Loss: 0.652939
Train Epoch: 125 [1536/2811 (55%)]	Loss: 0.402417
Train Epoch: 125 [2048/2811 (73%)]	Loss: 0.681670
Train Epoch: 125 [2560/2811 (91%)]	Loss: 0.475411
val set: Average loss: 0.5801, Accuracy: 554/704 (78.69%)

Train Epoch: 126 [0/2811 (0%)]	Loss: 0.481551
Train Epoch: 126 [512/2811 (18%)]	Loss: 0.638067
Train Epoch: 126 [1024/2811 (36%)]	Loss: 0.415908
Train Epoch: 126 [1536/2811 (55%)]	Loss: 0.463779
Train Epoch: 126 [2048/2811 (73%)]	Loss: 0.567509
Train Epoch: 126 [2560/2811 (91%)]	Loss: 0.475914
val set: Average loss: 0.6245, Accuracy: 555/704 (78.84%)

Train Epoch: 127 [0/2811 (0%)]	Loss: 0.247370
Train Epoch: 127 [512/2811 (18%)]	Loss: 0.470979
Train Epoch: 127 [1024/2811 (36%)]	Loss: 0.538718
Train Epoch: 127 [1536/2811 (55%)]	Loss: 0.408294
Train Epoch: 127 [2048/2811 (73%)]	Loss: 0.440663
Train Epoch: 127 [2560/2811 (91%)]	Loss: 0.489368
val set: Average loss: 0.6382, Accuracy: 551/704 (78.27%)

Train Epoch: 128 [0/2811 (0%)]	Loss: 0.791457
Train Epoch: 128 [512/2811 (18%)]	Loss: 0.414760
Train Epoch: 128 [1024/2811 (36%)]	Loss: 0.623057
Train Epoch: 128 [1536/2811 (55%)]	Loss: 0.516209
Train Epoch: 128 [2048/2811 (73%)]	Loss: 0.501235
Train Epoch: 128 [2560/2811 (91%)]	Loss: 0.546888
val set: Average loss: 0.6294, Accuracy: 542/704 (76.99%)

Train Epoch: 129 [0/2811 (0%)]	Loss: 0.463042
Train Epoch: 129 [512/2811 (18%)]	Loss: 0.771740
Train Epoch: 129 [1024/2811 (36%)]	Loss: 0.478656
Train Epoch: 129 [1536/2811 (55%)]	Loss: 0.577604
Train Epoch: 129 [2048/2811 (73%)]	Loss: 0.469122
Train Epoch: 129 [2560/2811 (91%)]	Loss: 0.502940
val set: Average loss: 0.7051, Accuracy: 539/704 (76.56%)

Train Epoch: 130 [0/2811 (0%)]	Loss: 0.417680
Train Epoch: 130 [512/2811 (18%)]	Loss: 0.533323
Train Epoch: 130 [1024/2811 (36%)]	Loss: 0.426028
Train Epoch: 130 [1536/2811 (55%)]	Loss: 0.736167
Train Epoch: 130 [2048/2811 (73%)]	Loss: 0.423383
Train Epoch: 130 [2560/2811 (91%)]	Loss: 0.538653
val set: Average loss: 0.6194, Accuracy: 551/704 (78.27%)

Train Epoch: 131 [0/2811 (0%)]	Loss: 0.497876
Train Epoch: 131 [512/2811 (18%)]	Loss: 0.473784
Train Epoch: 131 [1024/2811 (36%)]	Loss: 0.503366
Train Epoch: 131 [1536/2811 (55%)]	Loss: 0.337498
Train Epoch: 131 [2048/2811 (73%)]	Loss: 0.593240
Train Epoch: 131 [2560/2811 (91%)]	Loss: 0.613703
val set: Average loss: 0.6393, Accuracy: 547/704 (77.70%)

Train Epoch: 132 [0/2811 (0%)]	Loss: 0.437518
Train Epoch: 132 [512/2811 (18%)]	Loss: 0.419690
Train Epoch: 132 [1024/2811 (36%)]	Loss: 0.531206
Train Epoch: 132 [1536/2811 (55%)]	Loss: 0.399348
Train Epoch: 132 [2048/2811 (73%)]	Loss: 0.380386
Train Epoch: 132 [2560/2811 (91%)]	Loss: 0.531914
val set: Average loss: 0.6124, Accuracy: 555/704 (78.84%)

Train Epoch: 133 [0/2811 (0%)]	Loss: 0.469460
Train Epoch: 133 [512/2811 (18%)]	Loss: 0.445298
Train Epoch: 133 [1024/2811 (36%)]	Loss: 0.473536
Train Epoch: 133 [1536/2811 (55%)]	Loss: 0.435433
Train Epoch: 133 [2048/2811 (73%)]	Loss: 0.470356
Train Epoch: 133 [2560/2811 (91%)]	Loss: 0.747450
val set: Average loss: 0.6384, Accuracy: 545/704 (77.41%)

Train Epoch: 134 [0/2811 (0%)]	Loss: 0.386453
Train Epoch: 134 [512/2811 (18%)]	Loss: 0.448817
Train Epoch: 134 [1024/2811 (36%)]	Loss: 0.421339
Train Epoch: 134 [1536/2811 (55%)]	Loss: 0.460730
Train Epoch: 134 [2048/2811 (73%)]	Loss: 0.447135
Train Epoch: 134 [2560/2811 (91%)]	Loss: 0.531644
val set: Average loss: 0.7184, Accuracy: 533/704 (75.71%)

Train Epoch: 135 [0/2811 (0%)]	Loss: 0.671510
Train Epoch: 135 [512/2811 (18%)]	Loss: 0.608549
Train Epoch: 135 [1024/2811 (36%)]	Loss: 0.383133
Train Epoch: 135 [1536/2811 (55%)]	Loss: 0.468958
Train Epoch: 135 [2048/2811 (73%)]	Loss: 0.385323
Train Epoch: 135 [2560/2811 (91%)]	Loss: 0.837880
val set: Average loss: 0.7112, Accuracy: 531/704 (75.43%)

Train Epoch: 136 [0/2811 (0%)]	Loss: 0.429605
Train Epoch: 136 [512/2811 (18%)]	Loss: 0.577408
Train Epoch: 136 [1024/2811 (36%)]	Loss: 0.386748
Train Epoch: 136 [1536/2811 (55%)]	Loss: 0.520245
Train Epoch: 136 [2048/2811 (73%)]	Loss: 0.438088
Train Epoch: 136 [2560/2811 (91%)]	Loss: 0.478050
val set: Average loss: 0.6529, Accuracy: 551/704 (78.27%)

Train Epoch: 137 [0/2811 (0%)]	Loss: 0.501414
Train Epoch: 137 [512/2811 (18%)]	Loss: 0.533925
Train Epoch: 137 [1024/2811 (36%)]	Loss: 0.483589
Train Epoch: 137 [1536/2811 (55%)]	Loss: 0.748957
Train Epoch: 137 [2048/2811 (73%)]	Loss: 0.365656
Train Epoch: 137 [2560/2811 (91%)]	Loss: 0.432778
val set: Average loss: 0.6886, Accuracy: 536/704 (76.14%)

Train Epoch: 138 [0/2811 (0%)]	Loss: 0.269384
Train Epoch: 138 [512/2811 (18%)]	Loss: 0.338143
Train Epoch: 138 [1024/2811 (36%)]	Loss: 0.420844
Train Epoch: 138 [1536/2811 (55%)]	Loss: 0.506079
Train Epoch: 138 [2048/2811 (73%)]	Loss: 0.508240
Train Epoch: 138 [2560/2811 (91%)]	Loss: 0.361612
val set: Average loss: 0.5718, Accuracy: 564/704 (80.11%)

Train Epoch: 139 [0/2811 (0%)]	Loss: 0.404184
Train Epoch: 139 [512/2811 (18%)]	Loss: 0.398586
Train Epoch: 139 [1024/2811 (36%)]	Loss: 0.600626
Train Epoch: 139 [1536/2811 (55%)]	Loss: 0.593228
Train Epoch: 139 [2048/2811 (73%)]	Loss: 0.486773
Train Epoch: 139 [2560/2811 (91%)]	Loss: 0.395236
val set: Average loss: 0.6837, Accuracy: 535/704 (75.99%)

Train Epoch: 140 [0/2811 (0%)]	Loss: 0.360478
Train Epoch: 140 [512/2811 (18%)]	Loss: 0.627253
Train Epoch: 140 [1024/2811 (36%)]	Loss: 0.566426
Train Epoch: 140 [1536/2811 (55%)]	Loss: 0.570212
Train Epoch: 140 [2048/2811 (73%)]	Loss: 0.292298
Train Epoch: 140 [2560/2811 (91%)]	Loss: 0.301312
val set: Average loss: 0.6017, Accuracy: 559/704 (79.40%)

Train Epoch: 141 [0/2811 (0%)]	Loss: 0.714351
Train Epoch: 141 [512/2811 (18%)]	Loss: 0.323755
Train Epoch: 141 [1024/2811 (36%)]	Loss: 0.478026
Train Epoch: 141 [1536/2811 (55%)]	Loss: 0.492573
Train Epoch: 141 [2048/2811 (73%)]	Loss: 0.456469
Train Epoch: 141 [2560/2811 (91%)]	Loss: 0.338435
val set: Average loss: 0.5769, Accuracy: 566/704 (80.40%)

Train Epoch: 142 [0/2811 (0%)]	Loss: 0.498830
Train Epoch: 142 [512/2811 (18%)]	Loss: 0.496178
Train Epoch: 142 [1024/2811 (36%)]	Loss: 0.401728
Train Epoch: 142 [1536/2811 (55%)]	Loss: 0.546228
Train Epoch: 142 [2048/2811 (73%)]	Loss: 0.531198
Train Epoch: 142 [2560/2811 (91%)]	Loss: 0.542255
val set: Average loss: 0.6028, Accuracy: 558/704 (79.26%)

Train Epoch: 143 [0/2811 (0%)]	Loss: 0.435723
Train Epoch: 143 [512/2811 (18%)]	Loss: 0.499933
Train Epoch: 143 [1024/2811 (36%)]	Loss: 0.710904
Train Epoch: 143 [1536/2811 (55%)]	Loss: 0.431362
Train Epoch: 143 [2048/2811 (73%)]	Loss: 0.463116
Train Epoch: 143 [2560/2811 (91%)]	Loss: 0.493133
val set: Average loss: 0.6257, Accuracy: 554/704 (78.69%)

Train Epoch: 144 [0/2811 (0%)]	Loss: 0.292740
Train Epoch: 144 [512/2811 (18%)]	Loss: 0.559568
Train Epoch: 144 [1024/2811 (36%)]	Loss: 0.533325
Train Epoch: 144 [1536/2811 (55%)]	Loss: 0.536729
Train Epoch: 144 [2048/2811 (73%)]	Loss: 0.455759
Train Epoch: 144 [2560/2811 (91%)]	Loss: 0.450785
val set: Average loss: 0.5664, Accuracy: 575/704 (81.68%)

Train Epoch: 145 [0/2811 (0%)]	Loss: 0.623606
Train Epoch: 145 [512/2811 (18%)]	Loss: 0.416597
Train Epoch: 145 [1024/2811 (36%)]	Loss: 0.485598
Train Epoch: 145 [1536/2811 (55%)]	Loss: 0.532545
Train Epoch: 145 [2048/2811 (73%)]	Loss: 0.374064
Train Epoch: 145 [2560/2811 (91%)]	Loss: 0.515365
val set: Average loss: 0.5653, Accuracy: 566/704 (80.40%)

Train Epoch: 146 [0/2811 (0%)]	Loss: 0.400732
Train Epoch: 146 [512/2811 (18%)]	Loss: 0.489883
Train Epoch: 146 [1024/2811 (36%)]	Loss: 0.708112
Train Epoch: 146 [1536/2811 (55%)]	Loss: 0.376767
Train Epoch: 146 [2048/2811 (73%)]	Loss: 0.555931
Train Epoch: 146 [2560/2811 (91%)]	Loss: 0.405957
val set: Average loss: 0.6210, Accuracy: 561/704 (79.69%)

Train Epoch: 147 [0/2811 (0%)]	Loss: 0.478713
Train Epoch: 147 [512/2811 (18%)]	Loss: 0.516478
Train Epoch: 147 [1024/2811 (36%)]	Loss: 0.325388
Train Epoch: 147 [1536/2811 (55%)]	Loss: 0.556777
Train Epoch: 147 [2048/2811 (73%)]	Loss: 0.650920
Train Epoch: 147 [2560/2811 (91%)]	Loss: 0.410061
val set: Average loss: 0.5852, Accuracy: 569/704 (80.82%)

Train Epoch: 148 [0/2811 (0%)]	Loss: 0.388487
Train Epoch: 148 [512/2811 (18%)]	Loss: 0.474795
Train Epoch: 148 [1024/2811 (36%)]	Loss: 0.531705
Train Epoch: 148 [1536/2811 (55%)]	Loss: 0.567518
Train Epoch: 148 [2048/2811 (73%)]	Loss: 0.545771
Train Epoch: 148 [2560/2811 (91%)]	Loss: 0.353347
val set: Average loss: 0.5249, Accuracy: 586/704 (83.24%)

Train Epoch: 149 [0/2811 (0%)]	Loss: 0.446860
Train Epoch: 149 [512/2811 (18%)]	Loss: 0.361750
Train Epoch: 149 [1024/2811 (36%)]	Loss: 0.314059
Train Epoch: 149 [1536/2811 (55%)]	Loss: 0.652695
Train Epoch: 149 [2048/2811 (73%)]	Loss: 0.320839
Train Epoch: 149 [2560/2811 (91%)]	Loss: 0.474850
val set: Average loss: 0.5858, Accuracy: 563/704 (79.97%)

Train Epoch: 150 [0/2811 (0%)]	Loss: 0.453165
Train Epoch: 150 [512/2811 (18%)]	Loss: 0.308331
Train Epoch: 150 [1024/2811 (36%)]	Loss: 0.502295
Train Epoch: 150 [1536/2811 (55%)]	Loss: 0.585443
Train Epoch: 150 [2048/2811 (73%)]	Loss: 0.348658
Train Epoch: 150 [2560/2811 (91%)]	Loss: 0.406553
val set: Average loss: 0.5955, Accuracy: 561/704 (79.69%)

test set: Average loss: 0.7464, Accuracy: 516/694 (74.35%)
```