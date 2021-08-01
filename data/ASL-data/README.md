# American Sign Language Letters > v1
https://public.roboflow.ai/object-detection/american-sign-language-letters

Provided by [David Lee](https://www.linkedin.com/in/daviddaeshinlee/)
License: Public Domain

# Overview

The `American Sign Language Letters` dataset is an object detection dataset of each ASL letter with a bounding box. David Lee, a data scientist focused on accessibility, curated and released the dataset for public use.

![Example Image](https://blog.roboflow.com/content/images/2020/10/alphabet-intro.gif)

# Use Cases

One could build a model that reads letters in sign language. For example, Roboflow user David Lee wrote about how he made the model demonstrated above in this [blog post](https://blog.roboflow.com/computer-vision-american-sign-language/)

# Using this Dataset

Use the `fork` button to copy this dataset to your own Roboflow account and export it with new preprocessing settings, or additional augmentations to make your model generalize better. 

# About Roboflow

[Roboflow](https://roboflow.ai) makes managing, preprocessing, augmenting, and versioning datasets for computer vision seamless.

Developers build computer vision models faster and more accurately with Roboflow. 
#### [![Roboflow Workmark](https://i.imgur.com/WHFqYSJ.png =350x)](https://roboflow.ai)


American Sign Language Letters - v1 v1
==============================

This dataset was exported via roboflow.ai on October 25, 2020 at 2:20 AM GMT

It includes 1728 images.
Letters are annotated in Multi-Class Classification format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -5 and +5 degrees
* Random shear of between -5° to +5° horizontally and -5° to +5° vertically
* Random brigthness adjustment of between -25 and +25 percent
* Random Gaussian blur of between 0 and 1.25 pixels


