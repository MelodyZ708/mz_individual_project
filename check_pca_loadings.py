import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import sys
import glob
from PIL import Image
from torchvision.transforms import ToTensor
from sklearn.decomposition import PCA

sys.path.append('/vol/bitbucket/mz325/individual_project')
from como.utils.image_processing import CNNFeatureExtractor

img = Image.open(sorted(glob.glob('/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/*.png'))[0]).convert('RGB')
img_tensor = ToTensor()(img).unsqueeze(0).cuda()
extractor = CNNFeatureExtractor(target_channels=8, device='cuda:0')
cnn = extractor(img_tensor)[0, 3:, :, :].cpu().numpy()
C, H, W = cnn.shape

pca = PCA(n_components=3)
pca.fit(cnn.reshape(C, H * W).T)

print("PCA loadings (rows=PC1/2/3, cols=Ch1~Ch8):")
for i in range(3):
    weights = pca.components_[i]
    line = "  ".join([f"Ch{j+1}={w:+.3f}" for j, w in enumerate(weights)])
    print(f"  PC{i+1}: {line}")
    top = np.argsort(np.abs(weights))[::-1][:3]
    top_str = ", ".join([f"Ch{t+1}({weights[t]:+.3f})" for t in top])
    print(f"       Top contributors: {top_str}")
    print(f"       Variance explained: {pca.explained_variance_ratio_[i]*100:.1f}%")
    print()
