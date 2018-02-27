# Brian Hardenstein's entry for the Kaggle/Statoil Iceberg/Ship Classification Contest
## Contest Overview
Use data from Kaggle and Statoil to make a classifier that predicts if something is an iceberg or a ship.

<p align="center">
<img src="/imgs/report/vigilant-iceberg_explanation_graphic_2.png" width="512px"/>
</p>

**The data provided is:**
unique identifier, two bands of radar data in log format, satellite incidence angle, is_iceberg label

| id | band1 | band2 | inc_angle | is_iceberg |
|:--:|:-----:|:-----:|:----------|:----------:|
|'aaa'|[...]|[...]|32.403|1.0|
|'rd1'|[...]|[...]|43.279|0.0|
|'bnf'|[...]|[...]|40.923|1.0|
|'kr2'|[...]|[...]|NaN|0.0|

**Scoring:** Lowest log loss on predicted values on a validation set.

_Additionally because so much of Kaggle contests revolves around impractical ensembles of models an additional self-imposed constraint of using only a single model was followed. (Which may have been too restrictive.)_

**Example:** [Beluga 2nd Place Contestant](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/discussion/48294) used a weighted average of 100 ensembled N.N. models.

## Results

<p align="center">
<img src="/imgs/report/log_scores_a.png" width="512px"/>
</p>

###### **_Scores above 1.0 were clipped as those scores weren't remotely competitive and skewed the distribution even further. Then Log10 transformation of scores was performed because initial distribution was log normal._**

**Measured Accuracy:** >95%
**Measured Loss:** 0.1491
**Leaderboard Position:** 372/3,343 (top 11%)

This was my first Kaggle competition.

## Project Overview

#### Development Platform
* Used AWS p2.xlarge instance running Ubuntu 16.04 with Nvidia K80 GPU
  + Created custom AWS AMI image for faster deployment of spot instances
  + [AMI Details Here](https://pixelatedbrian.github.io/2018-01-12-AWS-Deep-Learning-with-GPU/)
  + Used Keras with Tensorflow CUDA/CUDNN GPU acceleration
* Annaconda Python 3.5
  + Used notebooks for EDA and prototyping
  + used python scripts in /src/ to actually run and evaluate models
  
#### Ideas Experimented and Evaluated

| Concept | Resulting Error | Integrated in Final Model |
|:--------------------------------------------|:---:|:---:|
|Transfer Learning: Inception V3|<img src="/imgs/report/up.png" width="20px"/>|<img src="/imgs/report/no.png" width="20px"/>|
|Transfer Learning: VGG 17|<img src="/imgs/report/up.png" width="20px"/>|<img src="/imgs/report/no.png" width="20px"/>|
|5 Layer ConvNet|<img src="/imgs/report/down.png" width="20px"/>|<img src="/imgs/report/yes.png" width="20px"/>|
|Shallow ConvNet|<img src="/imgs/report/up.png" width="20px"/>|<img src="/imgs/report/no.png" width="20px"/>|
|~10 Layer ConvNet|<img src="/imgs/report/up.png" width="20px"/>|<img src="/imgs/report/no.png" width="20px"/>|
|Standardize Data|<img src="/imgs/report/down.png" width="20px"/>|<img src="/imgs/report/yes.png" width="20px"/>|
|Concatenate inc_angle\*|<img src="/imgs/report/even.png" width="25px"/>|<img src="/imgs/report/no.png" width="20px"/>|
|Data Augmentation: Vertical Flip|<img src="/imgs/report/down.png" width="20px"/>|<img src="/imgs/report/yes.png" width="20px"/>|
|Data Augmentation: Horizontal Flip|<img src="/imgs/report/down.png" width="20px"/>|<img src="/imgs/report/yes.png" width="20px"/>|
|Data Augmentation: Orthagonal Rotation|<img src="/imgs/report/even.png" width="25px"/>|<img src="/imgs/report/no.png" width="20px"/>|
|Randomized Hyperparameter Search|<img src="/imgs/report/down.png" width="20px"/>|<img src="/imgs/report/yes.png" width="20px"/>|
|Early Stopping|<img src="/imgs/report/down.png" width="20px"/>|<img src="/imgs/report/yes.png" width="20px"/>|
|Signal Processing: Gaussian Smoothing|<img src="/imgs/report/up.png" width="20px"/>|<img src="/imgs/report/no.png" width="20px"/>|
|Signal Processing: Noise Crop|<img src="/imgs/report/up.png" width="20px"/>|<img src="/imgs/report/no.png" width="20px"/>|
|Signal Processing: Derivative|<img src="/imgs/report/even.png" width="25px"/>|<img src="/imgs/report/no.png" width="20px"/>|
|Drop inc_angle|<img src="/imgs/report/down.png" width="20px"/>|<img src="/imgs/report/yes.png" width="20px"/>|
|Drop 3rd 'Color' Channel|<img src="/imgs/report/down.png" width="20px"/>|<img src="/imgs/report/yes.png" width="20px"/>|
###### \* Initially concatenating inc_angle into the models seemed positive to neutral but later there was a large drop in error when inc_angle was not included in the data pipeline. It may be the flipping/flopping killed the signal that came from inc_angle. Other competitors also seemed to find that not including inc_angle helped with results.

#### Project Workflow:

<p align="center">
<img src="/imgs/report/model_flowchart_2.png" width="512px"/>
</p>

#### Data Overview:

<p align="center">
<img src="/imgs/report/statoil.jpg" width="512px"/>
</p>

Radar data was gathered by a satellite at an altitude of ~600km. The radar gathered information in 2 bands and also included an incidence angle as well as identifying id and also an is_iceberg label.

The radar data itself consisted of log values in a 2d matrix of shape=(75,75).  This can be thought of as an image but that is somewhat deceptive because the values originally ranged from roughly -30.0 to   +30.0.

As the above 3d images show even on ideal data the water was very noisy and choppy. Signal processing techniques to attempt to smooth or crop some of the noise increased the evaluation error.  One promising technique was taking the derivative of the contours, looking somewhat like edge detection. But at the time that was evaluated it was used as a 3rd color channel. The resulting error appeared to be even to slightly increased.

I hypothesize that if I had trained a model with the transformation applied to both channels and then used the resulting predictions in conjunction with the standard model's predictions (in ensemble) that the error would improve.  However because of the (artificial) single model constraint this was not attempted during the contest.

Alternatively, the transformed data could be added as 3rd and 4th channels but most likely the model complexity would need to be increased because the type of data and features that the model would need to generalize to would have increased quite a bit. I speculate that this is why having a 3rd channel of transformed data actually increased the evaluation error. (relative to the same architecture without the transformed data)
