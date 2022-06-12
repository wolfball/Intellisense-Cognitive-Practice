import os

basedir = 'experiments'
for filename in os.listdir(basedir):
    resultdirlist = os.listdir(os.path.join(basedir, filename))
    if 'resnet101_attention_b128_emd300_predictions.json' in resultdirlist and 'result.txt' not in resultdirlist:
        os.system("bash eval.sh " + filename)
