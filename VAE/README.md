### Commands for running

The following are the command for the best configuration, the result can be visualized by Tensorboard. The running time is about 1h.

```bash
python example.py \
--use_cuda 0 \  # specify the cuda id
--epoch 200 \
--lat_dim 100 \  # latent dimension (can get visualization if less than 3)
--en_layers 3 \  # number of encoder layers
--hid_dim 2000 \  # hiddent size
--beta 1.0 \  # beta for balancing the two parts in loss function
--noised \  # use noise augmentation
--expname best_exp

# Or
bash run.sh 0  # 0 is the cuda id which you can change
```

### Task summary

- **Requirements are FINISHED** Realize the VAE pipeline. Visualize the image when latent dimension is equal to 1 and 2. Adjust parameters to get the best result.
- **Requirements for report are SATISFIED** Formularize the principle of VAE, design experiments on different parameters, and add some interesting observation while adjusting parameters and changing the form of loss function.

### Some visualization

The best reconstructed result:

<img src="vis\best.png" alt="best" style="zoom: 67%;" />

The sample visualization for different latent dimensions:

<img src="vis\latdim_gene.png" alt="latdim_gene" style="zoom: 67%;" />

The latent space visualization for different latent dimensions:

<img src="vis\latdim_vis.png" alt="latdim_vis" style="zoom:67%;" />