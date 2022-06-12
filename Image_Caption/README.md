### Commands for running

For best configuration, the followings are the commands:

```bash
# First, you should change the "dataset_base_path" in configs/resnet101_attention.yaml
# to the path of data in your PC

# Then execute the training with GPU 0
# This model uses sinusoidal schedule sampling with k=5
bash run.sh 0

# Evaluate all experiments
python eva_exps.py
```

When first use SPICE, you should download some pkgs, which are avaliable in:

* /dssg/home/acct-stu/stu464/.conda/envs/pytorch/lib/python3.7/site-packages/pycocoevalcap/spice/spice-1.0.jar
* /dssg/home/acct-stu/stu464/.conda/envs/pytorch/lib/python3.7/site-packages/pycocoevalcap/spice/lib/stanford-corenlp-3.6.0.jar
* /dssg/home/acct-stu/stu464/.conda/envs/pytorch/lib/python3.7/site-packages/pycocoevalcap/spice/lib/stanford-corenlp-3.6.0-models.jar

### Task summary

- **Basic requirements are FINISHED:** Learn the different ways to process multi-modal data and the basic architecture of Encoder-Decoder model. Realize the schedule sampling. Understand the goals of different metrics and print them with the code.
- **High-level requirements are FINISHED:** Adjust the hyper parameters in schedule sampling, and presents a new decay schedule method which outperforms the traditional ones. Besides, I try to use pre-trained ViTExtractor to additionally process the image, but no improvement is seen (Add ViT model will be better but my gpu can not hold that memory).
- **Report requirements are FINISHED:** See Report.pdf. 