### Brian Hardenstein's entry for the Kaggle/Statoil Iceberg/Ship Classification Contest
### [pixelatedbrian](https://www.github.com/pixelatedbrian/vigilant-iceberg)

#### Contest Overview

The challenge is to use data from Kaggle and Statoil to make a classifier that predicts if something is an iceberg or a ship.  The data provided is two bands of radar data 75px * 75px as well as the satellite incidence angle.  Finally labels are provided in the training data in the form of an is_iceberg boolean.

The scoring of the contest was the lowest log error on predicted values on a validation set wins.

Additionally as so much of Kaggle contests seems to revolve around impractical ensembles of models I included an additional constraint in that I would take predictions from only a single model.  This was probably too rigid of a constraint but in a real scenario it would make deployment much more feasible.

Please see this post contest report by the second place contestant in which he used a weighted average of ~100 ensembled models. [Beluga 2nd Place] (https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/discussion/48294)

#### Results

On the final leader board I placed 372/3,343 teams. This was my first Kaggle competition and I learned a lot. I'm looking forward to my next challenge.

![competition results](/imgs/report/log_scores.png)
Scores above 1.0 were clipped as those scores weren't remotely competitive and skewed the distribution even further. Then Log10 transformation of scores was performed because initial distribution was log normal.
