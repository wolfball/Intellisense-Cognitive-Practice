### Commands for running

First, you should create a new environment according to **requirements.txt**. And put **data/gigaspeech** at the same directory with **main.py**.

Within 20 epochs, the best configuration is listed in the following:

```bash
# test ppl = 125.07, parameter size = 59.74M
python main.py --cuda --cuda_id 0 --emsize 800 --nhid 800 --nlayers 3  # Or bash run.sh
```

The followings are on GPT2 model, which will take about 6 hours for 10 epochs (overfit after 4 epochs). For faster training, you can change the model to DistilGPT2, which will reach 30.1 epochs after 9 epochs( about 3 hours).

```bash
# You can choose the output_dir by changing Line 50.
# test ppl = 28.2
python transformers_gpt2.py

# if you want to use distilgpt2,
# change the configuration in Line10 and Line47 from "gpt2" to "distilgpt2".
```

### Task summary

1. **Basic requirements are FINISHED** Have tried -- GRU，LSTM，Transformer -- three models，discuss and explore the impacts of hyper parameters, such as the number of layers, batch size, learning rate and so on. When adjusting hyper parameters, the parameter sizes of all models are less than 60M.
2. **Report is FINISHED** 
3. **Bonus items are FINISHED** Use Tensorboard to record the train and valid ppl during training phase；Utilize GPT2 to continuously improve the performance.
