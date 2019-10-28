# Controllable Text Attribute Transfer

Code for the paper: `Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation`[(Arxiv:1905.12926)](https://arxiv.org/abs/1905.12926)

## Introduction
In this work, we present a controllable unsupervised text attribute transfer framework, which can edit the entangled latent representation instead of modeling attribute and content separately. Specifically, we first propose a Transformer-based autoencoder to learn an entangled latent representation for a discrete text, then we transform the attribute transfer task to an optimization problem and propose the Fast-Gradient-Iterative-Modification algorithm to edit the latent representation until conforming to the target attribute. To the best of our knowledge, this is the first one that can not only control the degree of transfer freely but also perform sentiment transfer over multiple aspects at the same time. 

![Model architecture](/file/model.png)

## Documents

### Dependencies
	Python 3.6
	PyTorch 0.4
	
### Directory description

<pre><code>Root
├─data/*        Store the data files used by models.
├─method/*      Store the source code of models.
└─outputs/*     Store the final output of models.
</code></pre>

###  Data Preprocessing
In the data directory，run:

	python preprocessed_data.py 


### Model Training

To train the model, run in the method directory:

	python main.py 

After the training finished, you can specify the check point directory in `main.py`,

	args.if_load_from_checkpoint = True
	args.checkpoint_name = "xxx"

Or you can load my check-point file:
    
    '/save/1557667911' for yelp;
    '/save/1557668663' for amazon;
    '/save/1557891887' for captions;    
    
and run:

	python main.py 

## Example

Negative ->Positive:
<pre><code>Source:                 it is n’t terrible , but it is n’t very good either .
Human:                  it is n’t perfect , but it is very good .
Our model(w={1.0}):     it is n’t terrible , but it is n’t very good either .
Our model(w={2.0}):     it is n’t terrible , but it is n’t very good delicious either .
Our model(w={3.0}):     it is n’t terrible , but it is very good delicious either .
Our model(w={4.0}):     it is n’t terrible , but it is very good and delicious .
Our model(w={5.0}):     it is n’t terrible , but it is very good and delicious appetizer .
Our model(w={6.0}):     it is excellent , and it is very good and delicious well .
</code></pre>



## LICENSE

[MIT](./LICENSE)

## Citation

If you find our codes is helpful for your research, please kindly consider citing our paper:

<pre><code>@inproceedings{DBLP:journals/corr/abs-1905-12926,
  author    = {Ke Wang and Hang Hua and Xiaojun Wan},
  title     = {Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation},
  booktitle = {NeurIPS},
  year      = {2019}
}
</code></pre>




