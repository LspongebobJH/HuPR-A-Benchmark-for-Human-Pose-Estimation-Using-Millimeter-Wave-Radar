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