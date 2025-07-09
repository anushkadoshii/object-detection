# SSD: Single Shot Multibox Object Detector

**Original research paper:** https://arxiv.org/abs/1512.02325

**original Code:** https://github.com/weiliu89/caffe/tree/ssd

This is an implementation of a 2016 research paper by Wei Liu et al. 

---

## Download Datasets

**Download VOC2007 trainval and test**

``` sh data/scripts/VOC2007.sh ```

**Download VOC2012 trainval**

``` sh data/scripts/VOC2012.sh ```

## Training SSD

**Download VGG16 backbone**

```
mkdir weights
cd weight
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

Now, to train SSD:

``` python train.py ```

## Evaluation

To evaulate a trained network:

``` python eval.py ```

