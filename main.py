"""Run baseline model"""
import os
import math
import random
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator

from models.seq2seq import Seq2Seq
from utils.checkpoint import Chechpoint
from trainer.trainer import Trainer
from utils.scorer import BleuScorer
from utils.prepare import prepare_set
from evaluator.predictor import Predictor

from utils.verbaldataset import VerbalDataset
from utils.constants import (
    SEED, CUDA, CPU, PAD_TOKEN, RNN_NAME, CNN_NAME,
    TRANSFORMER_NAME, ATTENTION_1, ATTENTION_2, QUESTION, QUERY
)

DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)

def parse_args():
    """Add arguments to parser"""
    parser = argparse.ArgumentParser(description='Verbalization dataset baseline models.')
    parser.add_argument('--model', default=RNN_NAME, type=str,
                        choices=[RNN_NAME, CNN_NAME, TRANSFORMER_NAME], help='model to train the dataset')
    parser.add_argument('--input', default=QUESTION, type=str,
                        choices=[QUESTION], help='use question as input')
    parser.add_argument('--attention', default=ATTENTION_2, type=str,
                        choices=[ATTENTION_1, ATTENTION_2], help='attention layer for rnn model')
    parser.add_argument('--cover_entities', action='store_true', help='cover entities')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--epochs_num', default=30, type=int, help='number of epochs')
    args = parser.parse_args()
    return args

def set_SEED():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    """Main method to run the models"""
    set_SEED()
    args = parse_args()
    train = pd.read_csv('/content/UIT-ViCoV19QA_train.csv',na_filter=False,delimiter='|')
    val = pd.read_csv('/content/UIT-ViCoV19QA_val.csv',na_filter=False,delimiter='|')
    test = pd.read_csv('/content/UIT-ViCoV19QA_test.csv',na_filter=False,delimiter='|')
    dataset, src_vocab, trg_vocab, train_data, valid_data, test_data, train_m, val_m, test_m = prepare_data()
    print('--------------------------------')
    print(f'Model: {args.model}')
    print(f'Model input: {args.input}')
    if args.model == RNN_NAME:
        print(f'Attention: {args.attention}')
    print('--------------------------------')
    print(f'Batch: {args.batch_size}')
    print(f'Epochs: {args.epochs_num}')
    print('--------------------------------')
    
    if args.model == RNN_NAME and args.attention == ATTENTION_1:
        from models.rnn1 import Encoder, Decoder
    elif args.model == RNN_NAME and args.attention == ATTENTION_2:
        from models.rnn2 import Encoder, Decoder
    elif args.model == CNN_NAME:
        from models.cnn import Encoder, Decoder
    elif args.model == TRANSFORMER_NAME:
        from models.transformer import Encoder, Decoder, NoamOpt

    # create model
    encoder = Encoder(src_vocab, DEVICE)
    decoder = Decoder(trg_vocab, DEVICE)
    model = Seq2Seq(encoder, decoder, args.model).to(DEVICE)
    
    parameters_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters_num:,} trainable parameters')
    print('--------------------------------')

    # create optimizer
    if model.name == TRANSFORMER_NAME:
        # initialize model parameters with Glorot / fan_avg
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        optimizer = NoamOpt(torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        optimizer = optim.Adam(model.parameters())

    # define criterion
    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.stoi[PAD_TOKEN])

    # train data
    trainer = Trainer(optimizer, criterion, args.batch_size, DEVICE)
    trainer.train(model, train_data, valid_data, num_of_epochs=args.epochs_num)
    
    
    ###############checked################
    
    # load model
    model = Chechpoint.load(model)

    # generate test iterator
    valid_iterator, test_iterator = BucketIterator.splits(
                                        (valid_data, test_data),
                                        batch_size=args.batch_size,
                                        sort_within_batch=True if args.model == RNN_NAME else False,
                                        sort_key=lambda x: len(x.src),
                                        device=DEVICE)

    # evaluate model
    valid_loss = trainer.evaluator.evaluate(model, valid_iterator)
    test_loss = trainer.evaluator.evaluate(model, test_iterator)



    val_ref = [list(filter(None, np.delete(i,[0,1]))) for i in val.values]
    test_ref = [list(filter(None, np.delete(i,[0,1]))) for i in test.values]
    

    # calculate blue score for valid and test data
    new_val = prepare_set(val)
    new_test = prepare_set(test)
    
    predictor = Predictor(model, src_vocab, trg_vocab, DEVICE)
    #valid_scorer = BleuScorer()
    test_scorer = BleuScorer()
    
    #valid_scorer.data_score(new_val, predictor,path)
    test_scorer.data_score(new_test, predictor,path)
    
    r = {'ppl':[round(math.exp(test_loss),3)],
     'BLEU-1':[test_scorer.average_score()[0]*100],
     'BLEU-4':[test_scorer.average_score()[1]*100],
     'METEOR':[test_scorer.average_meteor_score()*100],
     'ROUGE-L':[test_scorer.average_rouge_score()*100]}
    df_result = pd.DataFrame(data=r)
    
#     print(f'| Val. Loss: {valid_loss:.3f} | Test PPL: {math.exp(valid_loss):7.3f} |')
#     print(f'| Val. Data Average BLEU score {valid_scorer.average_score()} |')
#     print(f'| Val. Data Average METEOR score {valid_scorer.average_meteor_score()} |')
#     print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
#     print(f'| Test Data Average BLEU score {test_scorer.average_score()} |')
#     print(f'| Test Data Average METEOR score {test_scorer.average_meteor_score()} |')



if __name__ == "__main__":
    # set a seed value
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    main()