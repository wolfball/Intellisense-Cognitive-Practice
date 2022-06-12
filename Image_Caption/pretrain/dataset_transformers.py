import glob
import ntpath
import io
import os

import nltk
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils_torch import split_data
from transformers import ViTFeatureExtractor, AutoTokenizer


class Flickr8kDataset(Dataset):
    """
    imgname: just image file name
    imgpath: full path to image file
    """

    def __init__(self, dataset_base_path='data/flickr8k/',
                 vocab_set=None, dist='val',
                 transformations=None,
                 return_raw=False,
                 load_img_to_memory=False,
                 return_type='tensor'):
        self.token = os.path.join(dataset_base_path, 'caption.txt')
        self.images_path = os.path.join(dataset_base_path, 'image')
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        # self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.dist_list = {
            'train': os.path.join(dataset_base_path, 'train_imgs.txt'),
            'val': os.path.join(dataset_base_path, 'val_imgs.txt'),
            'test': os.path.join(dataset_base_path, 'test_imgs.txt')
        }

        self.load_img_to_memory = load_img_to_memory
        self.pil_d = None

        self.return_raw = return_raw
        self.return_type = return_type

        self.__get_item__fn = self.__getitem__corpus if return_type == 'corpus' else self.__getitem__tensor

        self.imgpath_list = glob.glob(os.path.join(self.images_path, '*.jpg'))
        self.all_imgname_to_caplist = self.__all_imgname_to_caplist_dict()
        self.imgname_to_caplist = self.__get_imgname_to_caplist_dict(self.__get_imgpath_list(dist=dist))

        self.transformations = transformations if transformations is not None else transforms.Compose([
            transforms.ToTensor()
        ])

        self.startseq = self.tokenizer.bos_token.strip()
        self.endseq = self.tokenizer.pad_token.strip()
        self.unkseq = self.tokenizer.pad_token.strip()
        self.padseq = self.tokenizer.pad_token.strip()

        if vocab_set is None:
            self.vocab, self.word2idx, self.idx2word, self.max_len = self.__construct_vocab()
        else:
            self.vocab, self.word2idx, self.idx2word, self.max_len = vocab_set
        self.db = self.get_db()

    def __all_imgname_to_caplist_dict(self):
        captions = open(self.token, 'r').read().strip().split('\n')
        imgname_to_caplist = {}
        for i, row in enumerate(captions):
            row = row.split('\t')
            row[0] = row[0][:len(row[0]) - 2]  # filename#0 caption
            if row[0] in imgname_to_caplist:
                imgname_to_caplist[row[0]].append(row[1])
            else:
                imgname_to_caplist[row[0]] = [row[1]]
        return imgname_to_caplist

    def __get_imgname_to_caplist_dict(self, img_path_list):
        d = {}
        for i in img_path_list:
            img_name = ntpath.basename(i)
            if img_name in self.all_imgname_to_caplist:
                d[img_name] = self.all_imgname_to_caplist[img_name]
        return d

    def __get_imgpath_list(self, dist='val'):
        dist_images = set(open(self.dist_list[dist], 'r').read().strip().split('\n'))
        dist_imgpathlist = split_data(dist_images, img=self.imgpath_list)
        return dist_imgpathlist

    def __construct_vocab(self):
        return self.tokenizer.vocab,\
               self.tokenizer.convert_tokens_to_ids,\
               self.tokenizer.convert_ids_to_tokens,\
               self.tokenizer.model_max_length

    def get_vocab(self):
        return self.vocab, self.word2idx, self.idx2word, self.max_len

    def get_db(self):

        if self.load_img_to_memory:
            self.pil_d = {}
            for imgname in self.imgname_to_caplist.keys():
                self.pil_d[imgname] = Image.open(os.path.join(self.images_path, imgname)).convert('RGB')

        if self.return_type == 'corpus':
            df = []
            for imgname, caplist in self.imgname_to_caplist.items():
                cap_wordlist = []
                cap_lenlist = []
                for caption in caplist:
                    toks = nltk.word_tokenize(caption.lower())
                    cap_wordlist.append(toks)
                    cap_lenlist.append(len(toks))
                df.append([imgname, cap_wordlist, cap_lenlist])
            return df

        # ----- Forming a df to sample from ------
        l = ["image_id\tcaption\tcaption_length\n"]
        for imgname, caplist in self.imgname_to_caplist.items():
            for cap in caplist:
                l.append(
                    f"{imgname}\t"
                    f"{cap.lower()}\t"
                    f"{len(nltk.word_tokenize(cap.lower()))}\n")
        img_id_cap_str = ''.join(l)

        df = pd.read_csv(io.StringIO(img_id_cap_str), delimiter='\t')
        return df.to_numpy()

    @property
    def pad_value(self):
        return self.tokenizer.pad_token_id

    def __getitem__(self, index: int):
        return self.__get_item__fn(index)

    def __len__(self):
        return len(self.db)

    def get_image_captions(self, index: int):
        """
        :param index: [] index
        :returns: image_path, list_of_captions
        """
        imgname = self.db[index][0]
        return os.path.join(self.images_path, imgname), self.imgname_to_caplist[imgname]

    def __getitem__tensor(self, index: int):
        imgname = self.db[index][0]
        caption = self.db[index][1]
        capt_ln = self.db[index][2]
        cap_toks = [self.startseq] + nltk.word_tokenize(caption) + [self.endseq]
        img_tens = self.pil_d[imgname] if self.load_img_to_memory else Image.open(
            os.path.join(self.images_path, imgname)).convert('RGB')
        img_tens = self.transformations(img_tens)
        # img_tens = self.feature_extractor(images=img_tens, return_tensors='pt').pixel_values[0]
        cap_tens = torch.LongTensor(self.max_len).fill_(self.pad_value)
        cap_tens[:len(cap_toks)] = torch.LongTensor([self.word2idx(word) for word in cap_toks])
        return img_tens, cap_tens, len(cap_toks)

    def __getitem__corpus(self, index: int):
        imgname = self.db[index][0]
        cap_wordlist = self.db[index][1]
        cap_lenlist = self.db[index][2]
        img_tens = self.pil_d[imgname] if self.load_img_to_memory else Image.open(
            os.path.join(self.images_path, imgname)).convert('RGB')
        # img_tens = self.feature_extractor(images=img_tens, return_tensors='pt').pixel_values[0]
        img_tens = self.transformations(img_tens)
        return img_tens, cap_wordlist, cap_lenlist, imgname