# Image splicing detection - python

> These are steps to get you started with the project:

- download Anaconda Navigator, open spyder and run
basic &quot;Hello, world!&quot; program. If it doesn&#39;t work set
environment variables.
- download Milk Library [from .whl file]
- download Mahotas Library [from .whl file]
- download cv2 library [ direct pip ]
- Copy all code in folder where spyder is set.
- Download dataset and put them in folders Like
negatives and positives.
- Download glob library to access folders which
contains image.
- Download any package which is not already present
in anaconda environment.
- Rest is simple, code is self explanatory.

# Features:

> In this project, an improved image splicing detection is purposed which is
> based on global and local features of an image. 

- Let's get some local features using SIFT which is a local feature extraction method:

# SIFT
> A robust interest detector SIFT is
> applied which is tweaked with center of mass algorithm which localizes the spliced
> object and only nearest points are used concentrically with respect to coordinates of
> center of mass of given image.

-  Let's get some global features of an image:

# Zernike moments
> zernike will give measure about how the mass is distributed all over image.

# Local binary pattern 
> Local binary pattern will give measure of how many pixels represent a particular code.

# Haralick Features
> Haralick Features which is a combination of feature vector which provides 13 useful
statistical features.

# Methodology: 
- Effective morphology based image filtering techniques are used to reduce the noise and get prominent edge map. 
- Final feature vector by applying PCA which reduces dimention to a fixed component and final feature vector is 
feeded to SVM classifier for training model.
- N-fold cross validation is used to get minimally overfitted and
accurate model.
