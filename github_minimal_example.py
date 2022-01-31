from __future__ import print_function, division
#import ipdb
import os
#import tensorflow as tf
from numpy.lib.function_base import gradient
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad
from torch.utils.data import DataLoader
from torchvision import transforms#, utils
from catalyst import utils
from sklearn.impute import SimpleImputer
from nilearn import image
import nilearn
from nilearn import plotting
import os
import nibabel as nib
import glob
from PIL import Image
from torch.utils.data.dataset import Dataset
import sklearn as skl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import KFold
from freesurfer_stats import CorticalParcellationStats
from catalyst import dl
from sklearn.metrics import mean_squared_error
from captum.attr import visualization as viz
from captum.attr import (
    Saliency, 
    IntegratedGradients,
    NoiseTunnel,
    LayerGradCam, 
    FeatureAblation, 
    LayerActivation, 
    LayerAttribution
)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

#creating dataset: set filepath to where data is
image_list = glob.glob('normalized_T1.nii.gz')

images_as_np = []
#gets images finalized and preprocessed
for i in range(len(image_list)):
    single_image_path = image_list[i]
    image_name = str(single_image_path)
    #open that image
    img = nib.load(image_name)
    #convert to numpy and preprocess to get into tensor
    img = img.get_fdata().astype('float32')
    img = (img - img.min()) / (img.max() - img.min())
    #labels = volume_list_ordered[i].iloc[0]
    #img, temp = nobrainer.volume.apply_random_transform_scalar_labels(img, labels)
    new_img = np.zeros(img.shape)
    new_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
    new_img = torch.from_numpy(np.expand_dims(new_img, 0)).float()
    images_as_np.append(new_img)
    print("Shape of img after normalization: {}".format(new_img.shape))
#%%

transform = transforms.Compose([
    transforms.ToTensor()
])

#model and data class
#data class
class FreeSurferData(Dataset):
  def __init__(self, df):
    self.image_list = df.iloc[:, 0]
    self.data_len = len(self.image_list)
    self.volume_list = df.iloc[:, -1]

  def __getitem__(self, index):
    #get one image
    im_normal = self.image_list[index]
    #get output
    label = volume_list[index].values.squeeze().astype('float32')
    return im_normal, label

  def __len__(self):
    return self.data_len


#CNN model
batch_size = 16
input_shape = [batch_size, 1, 256, 256, 256]

def conv_pool(*args, **kwargs):
    """Configurable Conv block with Batchnorm and Dropout"""
    return nn.Sequential(
        nn.Conv3d(*args, **kwargs),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2),
    )

params = [
    {
        "in_channels": 1,
        "kernel_size": 11,
        "out_channels": 144,
        "stride": 3,
    },
    {
        "in_channels": 144,
        "kernel_size": 5,
        "out_channels": 192,
        "stride": 2,
        "bias": False,
    },
    {
        "in_channels": 192,
        "kernel_size": 5,
        "out_channels": 192,
        "stride": 1,
        "bias": False,
    },


    ]

class model(nn.Module):
    """Configurable Net from https://www.frontiersin.org/articles/10.3389/fneur.2020.00244/full"""

    def __init__(self, n_classes):
        """Init"""

        super(model, self).__init__()
        layers = [conv_pool(**block_kwargs) for block_kwargs in params]
        layers.append(nn.Dropout3d(.4))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=192, out_features=374))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(.4))
        layers.append(nn.Linear(in_features=374, out_features=374))
        layers.append(nn.Linear(in_features=374, out_features=n_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        #x = x.reshape(68)
        #print("OUTPUT SHAPE: {}".format(x.shape))
        return x


class ModelWrapper(torch.nn.Module):
    def __init__(self, module, i):
        super().__init__()
        self.module = module
        self.i = i
    def forward(self, x):
        y = self.module(x)
        #getting last batch of data
        y = y[-1]
        #reshaping data so that the attribution function can use the data
        y = torch.reshape(y, (1,68))
        return y

n_classes = 68
#creating splits for test and train data and kfold

def attribute_image_features(algorithm, model, features, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(
        features,
        **kwargs
    )
    
    return tensor_attributions

features = images_as_np[0]

features = torch.tensor(features).unsqueeze(0)
features.requires_grad = True


mymodel = model(n_classes)

logdir = "../logs/.../"
checkpoint = utils.load_checkpoint("train.30.pth")

utils.unpack_checkpoint(checkpoint, model=mymodel)
mymodel = ModelWrapper(mymodel, 0)

ig = IntegratedGradients(mymodel)
saliency = Saliency(mymodel)
nt = NoiseTunnel(saliency)

image_name = str(single_image_path)

#visualizing saliency
mymodel.zero_grad()
#getting attribution
attr_ig= saliency.attribute(
    features,
    target=i)

#getting the voxel value with the higest activation
#attr_ig = torch.load('attr_ig.pt')
arg_max = torch.argmax(attr_ig)

#index of maximum activation for a given brain region
idx = np.unravel_index(arg_max, attr_ig.shape)
idx = (idx[1], idx[2], idx[3])

print("Idx: {}".format(idx))

torch.save(features, 'features.pt')
torch.save(attr_ig, 'attr_ig.pt')

#creating two arrays that can be modified to get slices
attr_ig_viz = attr_ig.squeeze().cpu().detach().numpy()
features_viz = features.squeeze().cpu().detach().numpy()

# Organize the data for visualisation in the transversal plane
transversal_feat = features_viz[:, idx[1], :]
transversal_ig = attr_ig_viz[:, idx[1], :]

plt.imshow(transversal_feat)
c = plt.imshow(transversal_ig, cmap='hot', alpha=0.4)
plt.colorbar(c)
plt.savefig('imshowmethod_transversal_both.png')
plt.clf()

plt.imshow(transversal_ig, cmap='hot')
plt.colorbar()
plt.savefig('imshowmethod_transversal_ig.png')
plt.clf()

_ = viz.visualize_image_attr(
attr_ig_viz, #(256,256,256) is shape and is a numpy array?
features_viz, #(256,256,256) is shape and is a numpy array?
method='blended_heat_map',
cmap='hot',
show_colorbar=True,
sign='positive',
outlier_perc=1
)
print("Saving captum visualization with gridding as Original_viz.png")
plt.savefig("Original_viz.png")