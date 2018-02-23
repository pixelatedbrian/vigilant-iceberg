# Brian Hardenstein's entry for the Kaggle/Statoil Iceberg/Ship Classification Contest
## Contest Overview
Use data from Kaggle and Statoil to make a classifier that predicts if something is an iceberg or a ship.

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

**Example:** [Beluga 2nd Place Contestant](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/discussion/48294) used a weighted average of ~100 ensembled N.N. models.

## Results
![competition results](/imgs/report/log_scores.png)
###### **_Scores above 1.0 were clipped as those scores weren't remotely competitive and skewed the distribution even further. Then Log10 transformation of scores was performed because initial distribution was log normal._**

On the final leader board I placed 372nd out of 3,343 teams, or roughly top 11%. This was my first Kaggle competition.

* Used AWS p2.xlarge instance running Ubuntu 16.04 with Nvidia K80 GPU
..+ Created custom AWS AMI image for faster deployment of spot instances
..+ [AMI Details Here](https://pixelatedbrian.github.io/2018-01-12-AWS-Deep-Learning-with-GPU/)
* ~10 different convolutional networks evaluated
* 2 Transfer learning models evaluated, Inception v3 & VGG 17
..+ Experimentation found that transfer learning models massively overfit
..+ Hypothesize that the small 'image' size caused issues here
..+ Also the shapes/textures/features for the image nets really didn't apply to the data provided
..+ Finally as there were 2 channels of radar data a third channel had to be synthesized in order to pretend to be the 3rd color channel for the RGB ConvNets
* [~6 newly trained architectures evaluated](/src/model_zoo_v2.py)
..+ Final submission only used 2 channels of 'color'
..+ Training from scratch enabled the networks to learn specific features of the radar data
..+ Multiple transformations were attempted as well as blurring and noise cropping. The one with the most potential was a derivative edge detection algorithm.
..+ As the image transformations increased error hypothesize that a few networks trained on transformations individually with their predictions then ensembled would perform better.
..+ Custom models trained tended to take about 15min per run to train on AWS p2.xlarge
* Model 2 from model_zoo_v2.py ultimately had the best performance
..+ Used script to perform automated randomized hyperparameter grid search
..+ model2 eventually got a final leaderboard error of 0.1491


I learned a lot from this project and I'm looking forward to my next challenge.

#### Project Workflow:
![model flow](/imgs/report/model_flowchart.png)


#### Data Overview:

![data overview](/imgs/report/vigilant-iceberg_explanation_graphic_2.png)
Radar data was gathered by a satellite at an altitude of ~600km. The radar gathered information in 2 bands and also included an incidence angle as well as identifying id and also an is_iceberg label.

The radar data itself consisted of log values in a 2d matrix of shape=(75,75).  This can be thought of as an image but that is somewhat deceptive because the values originally ranged from roughly -30.0 to ..+30.0.
