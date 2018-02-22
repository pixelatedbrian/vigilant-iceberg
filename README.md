## Brian Hardenstein's entry for the Kaggle/Statoil Iceberg/Ship Classification Contest
[pixelatedbrian](https://www.github.com/pixelatedbrian/vigilant-iceberg)

### Contest Overview
====================

Use data from Kaggle and Statoil to make a classifier that predicts if something is an iceberg or a ship.

**The data provided is:**
* two bands of radar
* satellite incidence angle
* is_iceberg label
* unique identifier

Scoring: Lowest log loss on predicted values on a validation set.

Additionally because so much of Kaggle contests revolves around impractical ensembles of models an additional self-imposed constraint of using only a single model was followed. (Which may have been too restrictive.)

Example: Second place contestant used a weighted average of ~100 ensembled N.N. models. [Beluga 2nd Place](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/discussion/48294)

### Results
===========
![competition results](/imgs/report/log_scores.png)
Scores above 1.0 were clipped as those scores weren't remotely competitive and skewed the distribution even further. Then Log10 transformation of scores was performed because initial distribution was log normal.

On the final leader board I placed 372nd out of 3,343 teams, or roughly top 11%. This was my first Kaggle competition. Before this contest I had recently completed a DeepLearning.ai Coursera course by Andrew Ng on Convolutional Neural Networks. I'd like to take a moment to thank Professor Ng and his team for sharing their knowledge in a manner that was immediately useful.

I evaluated about 10 different convolutional networks, including a couple of instances of transfer learning. (Inception v3 and VGG 19) I primarily iterated on about 6 N.N.s that are found in this repo's /src/model_zoo.py.  Model 2 ultimately had the best performance and with extensive hyperparameter tuning was finally used for the final submissions which resulted in my final ranking.

For hardware I used a since AWS P2.xlarge instance running Ubuntu 16.04 with a NVIDIA K80 GPU. I created an AWS AMI image for this build for rapid deployment of spot instances, if you're interested please see [this](https://pixelatedbrian.github.io/2018-01-12-AWS-Deep-Learning-with-GPU/) blog post.    

I learned a lot from this project and I'm looking forward to my next challenge.

#### Project Workflow:
![model flow](/imgs/report/model_flowchart.png)


#### Data Overview:

![data overview](/imgs/report/vigilant-iceberg_explanation_graphic_2.png)
Radar data was gathered by a satellite at an altitude of ~600km. The radar gathered information in 2 bands and also included an incidence angle as well as identifying id and also an is_iceberg label.

The radar data itself consisted of log values in a 2d matrix of shape=(75,75).  This can be thought of as an image but that is somewhat deceptive because the values originally ranged from roughly -30.0 to +30.0.
