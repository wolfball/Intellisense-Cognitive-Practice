from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from datasets import load_dataset
import os
from PIL import Image

base_dir = "/data3/hyan/data/image_caption/"
# google/vit-base-patch16-224, ydshieh/vit-gpt2-coco-en
pretrained_dir ="nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(pretrained_dir)
feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_dir)
tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)


dataset = load_dataset("text",
                       data_files={"train": os.path.join(base_dir, "train_imgs.txt"),
                                   "validation": os.path.join(base_dir, "val_imgs.txt"),
                                   "test": os.path.join(base_dir, "test_imgs.txt"),})


def all_imgname_to_caplist_dict(token=os.path.join(base_dir, 'caption.txt')):
    captions = open(token, 'r').read().strip().split('\n')
    imgname_to_caplist = {}
    for i, row in enumerate(captions):
        row = row.split('\t')
        row[0] = row[0][:len(row[0]) - 2]  # filename#0 caption
        if row[0] in imgname_to_caplist:
            imgname_to_caplist[row[0]].append(row[1])
        else:
            imgname_to_caplist[row[0]] = [row[1]]
    return imgname_to_caplist

imgname2caplist = all_imgname_to_caplist_dict()

def process_function(example):
    result = {'caption': tokenizer(imgname2caplist[example['text']], truncation=True)}
    # result['pixelval'] = feature_extractor(images=Image.open(os.path.join(base_dir, example['text'])).convert(mode='RGB'),
    #                                       return_tensors='pt').pixel_values
    return result

dataset_pro = dataset.map(process_function)
print(dataset_pro['test'][0])

def extract_pixelval(example):
    # print(example.keys())
    images = [Image.open(os.path.join(base_dir, 'image', i)).convert(mode='RGB') for i in example['text']]
    result = {'pixelval': feature_extractor(images=images, return_tensors='pt').pixel_values}
    # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # total_length = len(concatenated_examples[list(examples.keys())[0]])
    # result = {
    #     k: [t[i * block_size: (i + 1) * block_size] for i in range(0, total_length // block_size - 1)]
    #     for k, t in concatenated_examples.items()
    # }
    # result["labels"] = result["input_ids"].copy()
    return result

import numpy as np
dataset_fin = dataset_pro.map(extract_pixelval, batched=True)
print(dataset_fin)
s = dataset_fin['test'][0]['pixelval']
print(np.array(s).shape)
print(type(s))

# def data_collator():
#
#
# from transformers import TrainingArguments
# training_args = TrainingArguments(output_dir="test_trainer",
#                                   evaluation_strategy="epoch",
#                                   learning_rate=2e-5,
#                                   weight_decay=0.01,
#                                   save_strategy="epoch",
#                                   logging_strategy="epoch",
#                                   num_train_epochs=10.0,
#                                   )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=lm_dataset["train"],
#     eval_dataset=lm_dataset["test"],
#     data_collator=data_collator,
# )