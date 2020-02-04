# [RSANet: Recurrent Slice-wise Attention Network for Multiple Sclerosis Lesion Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_46) (MICCAI'2019)
![Pytorch](https://img.shields.io/badge/Implemented%20in-Pytorch-red.svg) <br>

[Hang Zhang](https://tinymilky.github.io/), Jinwei Zhang, Qihao Zhang, Jeremy Kim, Shun Zhang, Susan A. Gauthier, Pascal Spincemaille, Thanh D. Nguyen, [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), and [Yi Wang](http://pre.weill.cornell.edu/mri/).

## Background

In this paper, we propose a novel recurrent slice-wise attention network (RSANet), which models 3D MRI images as sequences of slices and captures long-range dependencies through a recurrent manner to utilize global contextual information of MS lesions. 
Three major advantages of RSANet are listed as follows:
* Slice-wise attention (SA) block can help capture long-range dependencies among slices of any direction in 3D medical images;
* By recurrently aggregating information from SA blocks along different directions can help capture global conetextual dependency information in 3D medical images;
* Our recurrent mechanism makes the module GPU memory friendly and high computational efficient, which can be plugged into any existing 3D CNN structure with negligible cost. 

## Usage

The dataset used to verify the performance of the proposed method is unvailable per the policy of [Weill Cornell Medicine](https://weill.cornell.edu/). 
However, algorithms mentioned in the [MICCAI'2019 paper](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_46) are available in this repositorty.

We use a simple U-Net as backbone to show how our RSA block can be pugged into existing network. <br>
`./src/RSANet.py` contains unet structure with detailed module import from `./src/backbones/unet.py`. only three lines of code are needed to use RSA block from `./src/attModules/rsaModules.py`, as can be read from `./src/RSANet.py`.


## Slice-wise Attention (SA) Block

<div align=center><img width=75% src="/figs/sa_block.png"/></div>

SA block along one particular direction can be applied in your model based on your own needs. 

## Recurrent Slice-wise Attention (RSA) Block

<div align=center><img width=75% src="/figs/rsa_block.png"/></div>

RSA block can be considered as an approximation and regularization of [non-local neural networks](https://arxiv.org/abs/1711.07971) but with highly efficient memory and computation consumption.

## An Example of Information Propagation in RSA Block

<div align=center><img width=75% src="/figs/rsa_concept.png"/></div>

## Qualitative Results

We choose one slice from a testing image, and compare the qualitative results of different models with ground truth labels. As we can see from the following figrues, since both 3D U-Net and non-local network are not able to efficiently capture the long-range dependencies between MS lesions and brain structure, they suffer from an over-segmenting problem.

<div align=center>
  <img width=22% src="/figs/1_roi.png"/>
  <img width=22% src="/figs/2_rsa111.png"/>
  <img width=22% src="/figs/3_ncl010.png"/>
  <img width=22% src="/figs/4_ncl000.png"/>
</div>

From left to right are ground truth label, results of RSA-111, NCL-010 and 3D U-Net. (More details in the [paper](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_46))

## Citation
If you are inspired by [RSANet](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_46) or use our [code](https://github.com/tinymilky/RSANett), please cite:
```
@inproceedings{zhang2019rsanet,
  title = {RSANet: Recurrent Slice-Wise Attention Network for Multiple Sclerosis Lesion Segmentation},
  author = {Zhang, Hang and Zhang, Jinwei and Zhang, Qihao and Kim, Jeremy and Zhang, Shun and Gauthier, Susan A and Spincemaille, Pascal and Nguyen, Thanh D and Sabuncu, Mert and Wang, Yi},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages = {411--419},
  year = {2019},
  organization = {Springer}
}
```