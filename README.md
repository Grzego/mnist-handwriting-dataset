# Handwriting MNIST dataset

This repository contains code for training a model to generate this dataset. If you are interested in how this was created please look at [my page](grzego.github.io/...) where I described the process.

![](https://github.com/Grzego/mnist-handwriting-dataset/blob/master/images/handwriting-animation.gif)

## How to preview?

To preview simply run this command:

```
python show.py
```

## How to train model?

##### 1. Go to page: [https://edwin-de-jong.github.io/blog/mnist-sequence-data](https://edwin-de-jong.github.io/blog/mnist-sequence-data)

##### 2. Download `digit-images-thinned.tar.gz` to `data` directory

##### 3. Unpack it in `data` directory (you should end up with: `data/digit-images-thinned`)

##### 4. Run this command from `data` directory:
```
python prepare_data.py
```
This will create pickle file with thinned MNIST dataset.

##### 5. Now you can run training from base directory:

```
python train.py
```
You can run `tensorboard` for some visualizations.

##### 6. After training if over you need to convert thinned dataset to handwritten dataset:
```
python convert.py --model=path_to_model --save=mnist-handwritten
                          ^ex. summary/experiment-0/models/handwriting-model-0
```

#### 7. Preview dataset:
```
python show.py --dataset=mnist-handwritten.pkl
```
