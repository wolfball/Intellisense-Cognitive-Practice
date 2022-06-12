import os
from pathlib import Path
import pickle
import random

import yaml
import numpy as np
import fire
import pandas as pd
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.bleu.bleu import Bleu

from dataset_transformers import Flickr8kDataset
from utils.utils_torch import words_from_tensors_fn
from utils.util import get_logger, ptb_tokenize
from model_transformers import Captioner
from transformers import VisionEncoderDecoderModel


class Runner(object):
    """Main class to run experiments"""

    def __init__(self, seed=1, cudaid=0):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = "cpu"
        if torch.cuda.is_available():
            device = f"cuda:{cudaid}"
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.device = torch.device(device)

    def get_dataloaders(self, args):
        load_img_to_memory = args["load_img_to_memory"]
        dataset_base_path = args["dataset_base_path"]
        batch_size = args["train_args"]["batch_size"]
        print(dataset_base_path)
        train_set = Flickr8kDataset(
            dataset_base_path=dataset_base_path, dist='train',
            return_type='tensor', load_img_to_memory=load_img_to_memory)
        vocab_set = train_set.get_vocab()
        val_set = Flickr8kDataset(
            dataset_base_path=dataset_base_path, dist='val',
            vocab_set=vocab_set, return_type='corpus',
            load_img_to_memory=load_img_to_memory)
        test_set = Flickr8kDataset(
            dataset_base_path=dataset_base_path, dist='test',
            vocab_set=vocab_set, return_type='corpus',
            load_img_to_memory=load_img_to_memory)
        train_eval_set = Flickr8kDataset(
            dataset_base_path=dataset_base_path, dist='train',
            vocab_set=vocab_set, return_type='corpus',
            load_img_to_memory=load_img_to_memory)
        train_transformations = transforms.Compose([
            transforms.Resize(256),  # smaller edge of image resized to 256
            transforms.RandomCrop(256),  # get 256x256 crop from random location
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),  # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                                 (0.229, 0.224, 0.225))
        ])
        eval_transformations = transforms.Compose([
            transforms.Resize(256),  # smaller edge of image resized to 256
            transforms.CenterCrop(256),  # get 256x256 crop from random location
            transforms.ToTensor(),  # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                                 (0.229, 0.224, 0.225))
        ])

        train_set.transformations = train_transformations
        val_set.transformations = eval_transformations
        test_set.transformations = eval_transformations
        train_eval_set.transformations = eval_transformations
        eval_collate_fn = lambda batch: (
            torch.stack([x[0] for x in batch]),
            [x[1] for x in batch],
            [x[2] for x in batch],
            [x[3] for x in batch]
        )
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=1)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                                num_workers=1, collate_fn=eval_collate_fn)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                                 num_workers=1, collate_fn=eval_collate_fn)
        train_eval_loader = DataLoader(train_eval_set, batch_size=batch_size,
                                       shuffle=False, num_workers=1, collate_fn=eval_collate_fn)
        return {
            "train": train_loader,
            "train_eval": train_eval_loader,
            "val": val_loader,
            "test": test_loader
        }

    def train_model(self, train_loader, model, loss_fn, optimizer, desc='',
                    startseq_idx=0, pad_value=0, p=1):
        running_acc = 0.0
        running_loss = 0.0
        model.train()
        t = tqdm(iter(train_loader), desc=f'{desc}', leave=False)
        for batch_idx, batch in enumerate(t):
            images, captions, lengths = batch

            images = images.to(self.device)
            captions = captions.to(self.device)
            optimizer.zero_grad()

            scores, caps_sorted, decode_lengths, alphas, sort_ind = model(
                images, captions, lengths, startseq_idx, pad_value, p)

            # Since decoding starts with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths,
                                          batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths,
                                           batch_first=True)[0]

            loss = loss_fn(scores, targets)
            loss.backward()
            optimizer.step()

            correct = (torch.argmax(scores, dim=1) == targets).sum().float().item()
            running_acc += correct / targets.size(0)
            running_loss += loss.item()
            t.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': running_acc / (batch_idx + 1)}, refresh=True)
        t.close()

        return running_loss / len(train_loader)

    def evaluate_model(self, data_loader, model, scorers,
                       tensor_to_word_fn, word2idx, desc='', **sample_kwargs):
        model.eval()
        key_to_pred = {}
        key_to_refs = {}
        t = tqdm(iter(data_loader), desc=f'{desc}', leave=False)
        for batch_idx, batch in enumerate(t):
            images, captions, lengths, imgids = batch
            images = images.to(self.device)
            outputs = tensor_to_word_fn(model.sample(
                images, startseq_idx=word2idx('<|endoftext|>'),
                **sample_kwargs).cpu().numpy())
            for img_id, caption, prediction in zip(imgids, captions, outputs):
                key_to_pred[img_id] = [" ".join(prediction)]
                key_to_refs[img_id] = [" ".join(ref) for ref in caption]
            t.set_postfix({
                'batch': batch_idx,
            }, refresh=True)
        t.close()
        key_to_refs = ptb_tokenize(key_to_refs)
        key_to_pred = ptb_tokenize(key_to_pred)
        results = {}
        for scorer in scorers:
            score, scores = scorer.compute_score(key_to_refs, key_to_pred)
            results[scorer.method()] = score
        return results, key_to_refs, key_to_pred

    def train(self, config_file, **kwargs):
        with open(config_file) as reader:
            config = yaml.load(reader, Loader=yaml.FullLoader)
        args = dict(config, **kwargs)

        dataloaders = self.get_dataloaders(args)

        vocab_set = dataloaders["train"].dataset.get_vocab()
        vocab, word2idx, idx2word, max_len = vocab_set

        with open(args['vocab_path'], 'wb') as f:
            pickle.dump(vocab_set, f)
        vocab_size = len(vocab)

        Path(args["outputpath"]).mkdir(parents=True, exist_ok=True)
        logger = get_logger(Path(args["outputpath"]) / "train.log")

        model = Captioner(encoded_image_size=14,
                          encoder_dim=2048,
                          attention_dim=args['attention_dim'],
                          embed_dim=args['embedding_dim'],
                          decoder_dim=args['decoder_size'],
                          vocab_size=vocab_size).to(self.device)
        logger.info(model)
        model_path = os.path.join(args["outputpath"],
                                  f"{args['model']}_b{args['train_args']['batch_size']}_"
                                  f"emd{args['embedding_dim']}")

        pad_value = dataloaders["train"].dataset.pad_value
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_value).to(self.device)
        # cider_scorer = Cider()
        # spice_scorer = Spice()
        scorers = [Bleu()]
        tensor_to_word_fn = words_from_tensors_fn(idx2word=idx2word)
        params = model.parameters()
        optimizer = torch.optim.RMSprop(params=params,
                                        lr=float(args['train_args']['learning_rate']))

        train_loss_min = 100
        val_bleu_max = 0.0
        num_epochs = args["train_args"]["num_epochs"]
        k = args['sample_k']
        for epoch in range(num_epochs):
            if len(args['schedule_sampling']) == 0:
                p = 1
            elif args['schedule_sampling'] == 'linear':
                p = max(0.1, 1 - epoch / k)
            elif args['schedule_sampling'] == 'exp':
                p = k ** epoch
            elif args['schedule_sampling'] == 'sigmoid':
                p = k / (k + np.exp(epoch / k))
            elif args['schedule_sampling'] == 'cycle':
                p = 0.5 * (1 + np.cos(epoch * 2 * np.pi / k))
            elif args['schedule_sampling'] == 'cyclin':
                leni = num_epochs // int(k)
                p = max(0.1, 1 - (epoch % leni) / leni * 2)
            train_loss = self.train_model(desc=f'Epoch {epoch + 1}/{num_epochs}',
                                          model=model,
                                          optimizer=optimizer,
                                          loss_fn=loss_fn,
                                          train_loader=dataloaders["train"],
                                          startseq_idx=word2idx('<|endoftext|>'),
                                          pad_value=pad_value,
                                          p=p)
            with torch.no_grad():
                val_results = self.evaluate_model(
                    desc=f'Val eval: ', model=model,
                    scorers=scorers,
                    tensor_to_word_fn=tensor_to_word_fn,
                    word2idx=word2idx,
                    data_loader=dataloaders["val"],
                    **config["sample_args"])[0]
                val_bleu = val_results["Bleu"]
                msg = f"Epoch {epoch + 1}/{num_epochs}, train_loss: " \
                      f"{train_loss:.3f}, val_bleu: {val_bleu[3]:.3f}"
                logger.info(msg)
                state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_loss_latest': train_loss,
                    'val_bleu_latest': val_bleu[3],
                    'train_loss_min': min(train_loss, train_loss_min),
                    'val_bleu_max': max(val_bleu[3], val_bleu_max)
                }
                if train_loss < train_loss_min:
                    train_loss_min = train_loss
                    torch.save(state, '{}_best_train.pt'.format(model_path))
                if val_bleu[3] > val_bleu_max:
                    val_bleu_max = val_bleu[3]
                    torch.save(state, '{}_best_val.pt'.format(model_path))

        state = torch.load(f'{model_path}_best_val.pt', map_location="cpu")
        model.load_state_dict(state["state_dict"])

        scorers = [Bleu()]
        with torch.no_grad():
            model.eval()
            train_scores, _, _ = self.evaluate_model(desc=f'Train: ',
                                                     model=model,
                                                     scorers=scorers,
                                                     tensor_to_word_fn=tensor_to_word_fn,
                                                     word2idx=word2idx,
                                                     data_loader=dataloaders["train_eval"],
                                                     **config["sample_args"])
            val_scores, _, _ = self.evaluate_model(desc=f'Val: ',
                                                   model=model,
                                                   scorers=scorers,
                                                   tensor_to_word_fn=tensor_to_word_fn,
                                                   word2idx=word2idx,
                                                   data_loader=dataloaders["val"],
                                                   **config["sample_args"])
            test_scores, _, _ = self.evaluate_model(desc=f'Test: ',
                                                    model=model,
                                                    scorers=scorers,
                                                    tensor_to_word_fn=tensor_to_word_fn,
                                                    word2idx=word2idx,
                                                    data_loader=dataloaders["test"],
                                                    **config["sample_args"])
            logger.info("evaluation of the best validation performance model: ")
            for setname, result in zip(('train', 'val', 'test'),
                                       (train_scores, val_scores, test_scores)):
                bleu = result["Bleu"][3]
                logger.info(setname, end=' ')
                logger.info(f'Bleu: {bleu:.3f}', end=' ')
                logger.info("")

    def evaluate(self, config_file, **kwargs):
        with open(config_file) as reader:
            config = yaml.load(reader, Loader=yaml.FullLoader)
        args = dict(config, **kwargs)

        vocab_set = pickle.load(open(args['vocab_path'], "rb"))
        test_set = Flickr8kDataset(dataset_base_path=args['dataset_base_path'],
                                   dist='test', vocab_set=vocab_set,
                                   return_type='corpus',
                                   load_img_to_memory=args["load_img_to_memory"])
        vocab, word2idx, idx2word, max_len = vocab_set
        vocab_size = len(vocab)

        eval_transformations = transforms.Compose([
            transforms.Resize(256),  # smaller edge of image resized to 256
            transforms.CenterCrop(256),  # get 256x256 crop from random location
            transforms.ToTensor(),  # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                                 (0.229, 0.224, 0.225))
        ])

        test_set.transformations = eval_transformations
        eval_collate_fn = lambda batch: (
            torch.stack([x[0] for x in batch]),
            [x[1] for x in batch], [x[2] for x in batch], [x[3] for x in batch])
        test_loader = torch.utils.data.DataLoader(test_set, num_workers=1,
                                                  batch_size=1, shuffle=False, collate_fn=eval_collate_fn)

        model = Captioner(encoded_image_size=14, encoder_dim=2048,
                          attention_dim=args["attention_dim"],
                          embed_dim=args["embedding_dim"],
                          decoder_dim=args["decoder_size"],
                          vocab_size=vocab_size, train_embd=False)
        model_path = os.path.join(args["outputpath"],
                                  f"{args['model']}_b{args['train_args']['batch_size']}_"
                                  f"emd{args['embedding_dim']}")
        state = torch.load(f'{model_path}_best_val.pt', map_location="cpu")
        model.load_state_dict(state["state_dict"])
        model = model.to(self.device)

        scorers = [Cider(), Spice()]
        tensor_to_word_fn = words_from_tensors_fn(idx2word=idx2word)

        with torch.no_grad():
            model.eval()
            test_scores, key_to_refs, key_to_pred = self.evaluate_model(
                desc=f'Test: ', model=model, scorers=scorers,
                tensor_to_word_fn=tensor_to_word_fn, data_loader=test_loader,
                word2idx=word2idx, **config["sample_args"])
            out_data = []
            for img_id in key_to_refs.keys():
                out_data.append({
                    "img_id": img_id,
                    "reference": key_to_refs[img_id],
                    "prediction": key_to_pred[img_id]
                })
            out_df = pd.DataFrame(out_data)
            out_df.to_json("{}_predictions.json".format(model_path),
                           orient='records', indent=4)
            test_spider = (test_scores["CIDEr"] + test_scores["SPICE"]) / 2
            print('test', end=' ')
            print(f'SPIDEr: {test_spider:.3f}', end=' ')
            print()

    def train_evaluate(self, config_file, **kwargs):
        self.train(config_file, **kwargs)
        self.evaluate(config_file, **kwargs)


if __name__ == "__main__":
    fire.Fire(Runner)
