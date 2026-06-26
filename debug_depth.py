import cv2
import numpy as np

depth = cv2.imread("/home/melody/data/tum/sequence_video_0625/depth/206138.879785.png", cv2.IMREAD_ANYDEPTH)
print(f"dtype: {depth.dtype}")
print(f"min: {depth.min()}, max: {depth.max()}, mean: {depth.mean():.1f}")
print(f"shape: {depth.shape}")
