# CAHE

In humanity's ongoing quest to craft natural and realistic avatars within virtual environments, the generation of authentic eye gaze behaviors stands paramount. Eye gaze not only serves as a primary non-verbal communication cue, but it also reflects cognitive processes, intent, and attentiveness, making it a crucial element in ensuring immersive interactions. However, automatically generating these intricate gaze behaviors presents significant challenges. Traditional methods can be both time-consuming and lack the precision to align gaze behaviors with the intricate nuances of the environment in which the avatar resides. To overcome these challenges, we introduce a novel two-stage approach to generate context-aware head-and-eye motions across diverse scenes. By harnessing the capabilities of advanced diffusion models, our approach adeptly produces contextually appropriate eye gaze points, further leading to the generation of natural head-and-eye movements. Utilizing Head-Mounted Display (HMD) eye-tracking technology, we also present a comprehensive dataset, which captures human eye gaze behaviors in tandem with associated scene features. We show that our approach consistently delivers intuitive and lifelike head-and-eye motions and demonstrates superior performance in terms of motion fluidity, alignment with contextual cues, and overall user satisfaction.

# Dataset

To download the dataset, please check [our project page](https://sites.google.com/view/context-aware-generation).

# Dependencies

* Python 3.7
* PyTorch
* CUDA capable GPU (one is enough)

See `requirements.txt` for a full list of packages required.

# Usage

## Training

Code to train the model resides in `train` folder. To train the Fixation Diffuser and Head-Eye Diffuser, you have to change the path to load the `imgToFix_opt.txt` and `fixToGaze_opt.txt` files in `data_loaders/dataset.py`.

```shell
python -m train.train_CAHE --save_dir <Path/to/save/checkpoints/and/results> --dataset imgToFix
```

* Use `--dataset` to choose one of the dataset in two stages `{imgToFix, fixToGaze}` (`imgToFix` is default).
* Add `--train_platform_type {ClearmlPlatform, TensorboardPlatform}` to track results with either [ClearML](https://clear.ml/) or [Tensorboard](https://www.tensorflow.org/tensorboard).

## Testing

Code to sample resides in `sample` folder.

```shell
python -m sample.generate --model_path <path/to/checkpoint/file/to/be/sampled> --test_txt <path/to/a/text/file/lists/test/numbers> --cond_path <path/to/condition/data>
```

## Visualization

Code to visualize the generated fixation points.

```shell
python -m visualize.visual_fixation
```

# Acknowledgement

This code is standing on the shoulders of giants. We want to thank the following contributors that our code is based on:

[guided-diffusion](https://github.com/openai/guided-diffusion), [motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model).
