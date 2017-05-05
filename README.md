# YouTube-8M Video multiclassification
This repo is forked from https://github.com/google/youtube-8m, the youtube 8m start code. Contains code for training and evaluating machine learning
models over the [YouTube-8M](https://research.google.com/youtube8m/) dataset.

## Running on Local environment

Requirements

This repo requires Tensorflow 1.0.0. 

Complete instructions for downloading the dataset on the
[YouTube-8M website](https://research.google.com/youtube8m/download.html).

video-level database is much smaller, only about 31 Gb. Frame-level database is large, which is about 1.71 Tb. 

### Using Video-Level Features

####  Training on Video-Level Features

To start training a logistic model on the video-level features, run

```sh
python train.py --train_data_pattern='/path/to/video_level_train_database/train*.tfrecord' --model=LogisticModel --train_dir=./video_model_save_path/ 
--feature_names="mean_rgb, mean_audio" --feature_sizes="1024, 128"
```

Since the dataset is sharded into 4096 individual files, we use a wildcard (\*)
to represent all of those files.

By default, the training code will frequently write _checkpoint_ files (i.e.
values of all trainable parameters, at the current training iteration). These
will be written to the `--train_dir`. If you re-use a `--train_dir`, the trainer will first restore the latest checkpoint written in that directory. This only
works if the architecture of the checkpoint matches the graph created by the
training code. 
Add `--start_new_model` at the end will start a new model and delete past model in this directory. 

#### Evaluation and Inference

To evaluate the model, run

```sh
python eval.py --eval_data_pattern='/path/to/features/validate*.tfrecord' --model=LogisticModel --train_dir=./video_model_save_path/ --run_once=True --feature_names="mean_rgb,mean_audio" --feature_sizes="1024,128"
```

As the model is training or evaluating, view the results on tensorboard
by running

```sh
tensorboard --logdir=./video_model_save_path/
```

and navigating to http://localhost:6006 in your web browser.

```sh
python inference.py --output_file=./video_model_save_path/predictions.csv --input_data_pattern='/path/to/features/test*.tfrecord' --train_dir=./video_model_save_path/
```

This will output the top 20 predicted labels from the model for every example
to 'predictions.csv'.

### Using Frame-Level Features

Follow the same instructions as above, appending
`--frame_features=True --model=FrameLevelLogisticModel --feature_names="rgb"
--feature_sizes="1024" --train_dir=$MODEL_DIR/frame_level_logistic_model`
for the 'train.py', 'eval.py', and 'inference.py' scripts.

The 'FrameLevelLogisticModel' is designed to provide equivalent results to a
logistic model trained over the video-level features. Please look at the
'models.py' file to see how to implement your own models.

### Using Audio Features

The feature files (both Frame-Level and Video-Level) contain two sets of
features: 1) visual and 2) audio. The code defaults to using the visual
features only, but it is possible to use audio features instead of (or besides)
visual features. To specify the (combination of) features to use you must set
`--feature_names` and `--feature_sizes` flags. The visual and audio features are
called 'rgb' and 'audio' and have 1024 and 128 dimensions, respectively.
The two flags take a comma-separated list of values in string. For example, to
use audio-visual Video-Level features the flags must be set as follows:

```
--feature_names="mean_rgb, mean_audio" --feature_sizes="1024, 128"
```

Similarly, to use audio-visual Frame-Level features use:

```
--feature_names="rgb, audio" --feature_sizes="1024, 128"
```

**NOTE:** Make sure the set of features and the order in which the appear in the
lists provided to the two flags above match. Also, the order must match when
running training, evaluation, or inference.

## Overview of Models

### Video-Level Models
*   `LogisticModel`: Linear projection of the output features into the label
                     space, followed by a sigmoid function to convert logit
                     values to probabilities.

### Frame-Level Models
* `LstmModel`: Processes the features for each frame using a multi-layered
               LSTM neural net. The final internal state of the LSTM
               is input to a video-level model for classification. 
* `FrameLevelLogisticModel`: Quite like 'LogisticModel', but apply average pooling before input data into full connected network. 

## Overview of Files
Most files are derivated from original youtube-8m startcode

### Training
*   `train.py`: The primary script for training models.
*   `losses.py`: Contains definitions for loss functions.
*   `models.py`: Contains the base class for defining a model.
*   `video_level_models.py`: Contains definitions for models that take
                             aggregated features as input.
*   `frame_level_models.py`: Contains definitions for models that take frame-
                             level features as input.
*   `model_util.py`: Contains functions that are of general utility for
                     implementing models.
*   `export_model.py`: Provides a class to export a model during training
                       for later use in batch prediction.
*   `readers.py`: Contains definitions for the Video dataset and Frame
                  dataset readers.

### Evaluation
*   `eval.py`: The primary script for evaluating models.
*   `eval_util.py`: Provides a class that calculates all evaluation metrics.
*   `average_precision_calculator.py`: Functions for calculating
                                       average precision.
*   `mean_average_precision_calculator.py`: Functions for calculating mean
                                            average precision.

### Inference
*   `inference.py`: Generates an output file containing predictions of
                    the model over a set of videos.

### Misc
*   `README.md`: This documentation.
*   `utils.py`: Common functions.
*   `convert_prediction_from_json_to_csv.py`: Converts the JSON output of
        batch prediction into a CSV file for submission.

