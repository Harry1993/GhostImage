# GhostImage

This repository demonstrates the emulation of our
GhostImage attacks [[PDF](http://u.arizona.edu/~yman/papers/ghostimage_raid20.pdf)]:

```
@inproceedings{man2020ghostimage,
  title={GhostImage: Remote Perception Domain Attacks against Camera-based
         Image Classification Systems},
  author={Man, Yanmao and Li, Ming and Gerdes, Ryan},
  booktitle={Proceedings of the 23rd International Symposium on Research in
             Attacks, Intrusions and Defenses (USENIX RAID 2020))},
  year={2020}
}
```

The main structure of our implementaion follows Nicolas Carlini's
[nn_robust_attacks](https://github.com/carlini/nn_robust_attacks).

## Pre-requisites

All scripts are written in Python 3, with dependencies
```
pillow == 7.1.2
numpy == 1.18.5
tensorflow == 2.1.0
tensorflow-probability == 0.10.0
```

### Pre-trained models

First of all, create a directory named `models`.

To download the [Inception
V3](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image)
pre-trained model:

```
$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
tar -C ./models -xz
```

To train a CIFAR model, use [this
script](https://github.com/carlini/nn_robust_attacks/blob/master/train_models.py)
written by Carlini. The network architecture would be the same as the one
defined in `setup_cifar.py`.

Download a pre-trained LISA classifier
[here](http://www2.engr.arizona.edu/~yman/ghostimage/lisa), and put it under
`models`.

## Attack Examples

Script `test_attack.py` presents an example function `single_image`,
demonstrating how we can invoke our attack core module, which is defined in
`rgb_attack.py`.

In the "main function" of `test_attack.py`, there are four attack examples:

### Alteration attack on ImageNet (Inception V3)

```
single_image(img_path='./benign_images/ILSVRC2012_val_00019992.JPEG',
             target_label=555, objective='alteration', dataset='imagenet',
             dig=.7, ana=.1, num_rows=20, num_columns=20)
```

