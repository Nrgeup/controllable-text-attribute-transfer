# Controllable Unsupervised Text Attribute Transfer

Code for the paper: `Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation`[(Arxiv:1905.12926)](https://arxiv.org/abs/1905.12926)

## Introduction
In this work, we present a controllable unsupervised text attribute transfer framework, which can edit the entangled latent representation instead of modeling attribute and content separately. Specifically, we first propose a Transformer-based autoencoder to learn an entangled latent representation for a discrete text, then we transform the attribute transfer task to an optimization problem and propose the Fast-Gradient-Iterative-Modification algorithm to edit the latent representation until conforming to the target attribute. To the best of our knowledge, this is the first one that can not only control the degree of transfer freely but also perform sentiment transfer over multiple aspects at the same time. 

![Model architecture](/file/model.png)

## Documents

### Dependencies
	Python3.6
	Pytorch0.4

###  Data preprocessing
In the data directory，run:

	python preprocessed_data.py 


More code is being sorted out！






