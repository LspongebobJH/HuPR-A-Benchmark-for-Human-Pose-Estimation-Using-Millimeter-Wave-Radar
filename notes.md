VRDAEmaps_hori.shape
[32, 8, 8, 2, 64, 64, 8]
number of samples: 32
adjacent frames for temporal modeling: 8
number of chirps with velocity information: 8
real and image parts: 2
range/azimute: 64
elevation: 8

Note: 
azimute (8) -> padding -> (64)
elevation (2) -> padding -> (8)
chirps (256) -> preprocessing -> (16) -> sample -> (8)

video/0/: video of skeleton prediction based on default experiment
video/1/: video of groud truth skeleton based on default exp

test: debug logs, no use
hupr1: the first successful attempt of distributed training on single-radar version of HuPR. Results show that the training is incomplete within 30 epochs. The training would be continued with more 30 epochs in hupr1-1
hupr1-1: continue the training of hupr1 with more 30 epochs. 