# 2020-12-14 - Jon Matthis
# pulling apart DeepPoseKit example iPython Notebooks into python scripts

import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from deepposekit.io import VideoReader, DataGenerator, initialize_dataset
from deepposekit.annotate import KMeansSampler
import tqdm
import glob
import pandas as pd
from pathlib import Path

print('Helo wrld')

debug = True

## A note on image resolutions
## Currently DeepPoseKit only supports image resolutions that can be repeatedly divided by 2. For example, all of these values are valid image resolutions for either height or width:
# exp = np.arange(1,12)
# exp = 2**exp

# print(1*exp)
# print(3*exp)
# print(5*exp)
# print(7*exp)
# print(11*exp)

# Sample video frames - This loads batches of 100 frames from the video, and then randomly samples frames from the batches to hold them in memory. You can use any method for sampling frames.

vidPath = Path(r'C:\Users\jonma\Dropbox\GitKrakenRepos\DeepPoseKit\poo_cat\Poo_CatVid.mp4') #current code can't handle path objects because it sucks



reader = VideoReader(vidPath.as_posix(), batch_size=10, gray=True)

if debug:
    testframe = reader[0] # read a frame
    plt.figure(figsize=(5,5))
    plt.imshow(testframe[0,...,0])
    plt.show()

randomly_sampled_frames = []
for idx in tqdm.tqdm(range(len(reader)-1)):
    batch = reader[idx]
    random_sample = batch[np.random.choice(batch.shape[0], 10, replace=False)]
    randomly_sampled_frames.append(random_sample) 
reader.close()

randomly_sampled_frames = np.concatenate(randomly_sampled_frames)
randomly_sampled_frames.shape

#Apply k-means to reduce correlation - This applies the k-means algorithm to the images using KMeansSampler to even out sampling across the distribution of images and reduce correlation within the annotation set.

kmeans = KMeansSampler(n_clusters=10, max_iter=1000, n_init=10, batch_size=100, verbose=True)
kmeans.fit(randomly_sampled_frames)

if debug:
    kmeans.plot_centers(n_rows=2)
    plt.show()

kmeans_sampled_frames, kmeans_cluster_labels = kmeans.sample_data(randomly_sampled_frames, n_samples_per_label=10)
kmeans_sampled_frames.shape

# Initialize a new data set for annotations - You can use any method for sampling images to create a numpy array with the shape (n_images, height, width, channels) and then initialize an annotation set. Check the doc string for more details:

initialize_dataset(
    images=kmeans_sampled_frames,
    datapath=r"C:\Users\jonma\Dropbox\Eyetracking\PupilLabs\PandEyeTracker_2020-06-20\PandaFace_skel.h5",
    skeleton=r"C:\Users\jonma\Dropbox\Eyetracking\PupilLabs\PandEyeTracker_2020-06-20\pandaFace_skel.csv",    # overwrite=True # This overwrites the existing datapath
)

# Create a data generator - This creates a DataGenerator for loading annotated data. Indexing the generator returns an image-keypoints pair, which you can then visualize. Right now all the keypoints are set to zero, because they haven't been annotated. - You can also look at the doc string for more explanation:

data_generator = DataGenerator(r"C:\Users\jonma\Dropbox\Eyetracking\PupilLabs\PandEyeTracker_2020-06-20\PandaFace_skel.h5", mode="full")

if debug:
    image, keypoints = data_generator[0]

    plt.figure(figsize=(5,5))
    image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
    cmap = None if image.shape[-1] is 3 else 'gray'
    plt.imshow(image, cmap=cmap, interpolation='none')
    for idx, jdx in enumerate(data_generator.graph):
        if jdx > -1:
            plt.plot(
                [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
                [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
                'r-'
            )
    plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50, cmap=plt.cm.hsv, zorder=3)

    plt.show()
    