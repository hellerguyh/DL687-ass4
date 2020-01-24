import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random as r

from collections import Counter

from torchtext import datasets
from torchtext import data
from torchtext import vocab

import json

import sys

FOLDER_PATH = "./gdrive/My Drive/Master/DL687/As4/1606.01933/data/snli_1.0/"
DEBUG = True


def DEBUG_PRINT(x):
    if DEBUG:
        print(x)

deviceCuda = torch.device("cuda")
deviceCPU = torch.device("cpu")
USE_CUDA = True
USE_840 = True
RUNNING_LOCAL = True 
if RUNNING_LOCAL:
    FOLDER_PATH = './data/snli_1.0/'
    USE_CUDA = False
#if USE_840:
#  print(FOLDER_PATH+'glove_cache')
#  GLOVE_DATA = GloVe(name='840B', dim=300, cache=FOLDER_PATH+'glove_cache/')
#else:
#  GLOVE_DATA = GloVe(name='6B', dim=300, cache=FOLDER_PATH+'glove_cache/')
GLOVE_VECTORS = vocab.GloVe(name='840B', dim=300, cache=FOLDER_PATH+'glove_cache/')
#GLOVE_COUNTER = Counter(list(GLOVE_VOCAB.stoi.keys()))

if USE_CUDA:
    USED_DEVICE = deviceCuda

def list2dict(lst):
    it = iter(lst)
    indexes = range(len(lst))
    res_dct = dict(zip(it, indexes))
    return res_dct


def reverseDict(d):
    vals = [''] * len(d.keys())
    for k in d.keys():
        vals[d[k]] = k
    return vals

class DB(object):
    def __init__(self, batch_size):
        self.data_field = data.Field(init_token='NULL', tokenize='spacy', batch_first=True, include_lengths=True)
        self.label_field = data.Field(sequential=False, batch_first=True)
      
        self.label_field.build_vocab([['contradiction'], ['entailment'], ['neutral']])

        self.train_ds, self.dev_ds, self.test_ds = datasets.SNLI.splits(self.data_field, self.label_field, root=FOLDER_PATH)

        self.data_field.build_vocab(self.train_ds, self.dev_ds, self.test_ds, vectors=GLOVE_VECTORS)
        
        '''fake_field = data.Field(tokenize='spacy')
        fake_field.build_vocab(self.train_ds, self.dev_ds, self.test_ds)
        fake_keys = fake_field.vocab.itos
        print(fake_keys)'''

        from collections import Counter

        fake_keys = Counter(list(self.data_field.vocab.stoi.keys()))
        self.glove_keys = [[key] for key in GLOVE_VECTORS.stoi.keys() if fake_keys[key] > 0]
        self.data_field.build_vocab(self.glove_keys, vectors=GLOVE_VECTORS)
        fake_keys = []

        self.train_iter, self.dev_iter, self.test_iter = data.BucketIterator.splits((self.train_ds, self.dev_ds, self.test_ds), 
                batch_size=batch_size, device=deviceCPU, sort_key=lambda d: len(d.premise), shuffle=False, sort=False)

    def getIter(self, iter_type):      
        if iter_type == "train":
            return self.train_iter
        elif iter_type == "dev":
            return self.dev_iter
        elif iter_type == "test":
            return self.test_iter
        else:
            raise Exception("Invalid type")

'''
if __name__ == '__main__':
    print('Hello world')
    db = DB(2)
    train_iter = db.getIter('train')
    wembeddings = nn.Embedding.from_pretrained(embeddings=db.data_field.vocab.vectors, freeze=True,
                                                    padding_idx=1)
    for sample_b in train_iter:
        premise, _ = sample_b.premise
        print(premise)
        print(wembeddings(premise[0]))
        raise Exception()
'''

class Tagger(nn.Module):
    def __init__(self, embedding_dim, projected_dim, tagset_size,
                 vectors, f_dim=200, v_dim=200, dropout=False):
        super(Tagger, self).__init__()
        self.embedding_dim = embedding_dim

        # Creat Embeddings
        vecs =vectors 
        vecs = vecs/torch.norm(vecs, dim=1, keepdim=True)
        ## Add to glove vectors 2 vectors for unknown and padding:
        for i in range(100):
            #pad = torch.rand((1, vecs[0].shape[0]))
            pad = torch.normal(mean=torch.zeros(1, vecs[0].shape[0]), std=1)
            vecs = torch.cat((vecs, pad), 0)
        pad = torch.zeros((1, vecs[0].shape[0]))
        vecs = torch.cat((vecs, pad), 0)
        vecs[1] = torch.zeros(vecs[0].shape)
        vecs[0] = torch.zeros(vecs[0].shape)
        self.wembeddings = nn.Embedding.from_pretrained(embeddings=vecs, freeze=True)
        ## project down the vectors to 200dim
        self.project = nn.Linear(embedding_dim, projected_dim)
        self.G = self.feedForward(f_dim * 2, v_dim, 0.2)
        self.H = self.feedForward(v_dim * 2, v_dim, 0.2)
        self.linear = nn.Linear(v_dim, tagset_size)
        self.hidden_dim = projected_dim
        self.f_dim = f_dim
        
        self.F = self.feedForward(self.hidden_dim, f_dim, 0.2)
        self.softmax = nn.Softmax(dim=1)

    def feedForward(self, i_dim, o_dim, dropout):
        use_dropout = dropout > 0
        layers = []

        layers.append(nn.Linear(i_dim, o_dim))
        if use_dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(o_dim, o_dim))
        if use_dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

        layers = nn.Sequential(*layers)
        return layers

    def forward(self, premise_data, hyp_data):
        #padded_premise_w = premise_data['w_data']
        #premise_w_lens = premise_data['w_lens']

        #padded_hyp_w = hyp_data['w_data']
        #hyp_w_lens = hyp_data['w_lens']

        if USE_CUDA:
            padded_premise_w = premise_data.to(deviceCuda)
            padded_hyp_w = hyp_data.to(deviceCuda)
        else:
            padded_premise_w = premise_data
            padded_hyp_w = hyp_data


        prem_w_e = self.wembeddings(padded_premise_w)
        hyp_w_e = self.wembeddings(padded_hyp_w)

        # Project the embeddings to smaller vector
        prem_w_e = self.project(prem_w_e)
        hyp_w_e = self.project(hyp_w_e)

        #beta, alpha = self.attention(prem_w_e, hyp_w_e)
        a = prem_w_e 
        b = hyp_w_e
        fa = self.F(a)
        fb = self.F(b)

        # We want ato calculate e_ij = fa_i * fb_j
        # fa shape: batch x sentence_a x hidden_dim
        # fb shape: batch x sentence_b x hidden_dim
        ## Per batch calculation:
        ## calc fa x fb.transpose() gives sentence_a x sentence_b
        E = torch.bmm(fa, torch.transpose(fb, 1, 2))

        # E shape: batch x sentence_a x sentence_b
        ## for beta needs: (batch*sentence_a)*sentence_b
        E4beta = self.softmax(E.view(-1, b.shape[1]))
        E4beta = E4beta.view(E.shape)
        beta = torch.bmm(E4beta, b)

        E4alpha = torch.transpose(E, 1, 2)
        saved_shape = E4alpha.shape
        E4alpha = self.softmax(E4alpha.reshape(-1, a.shape[1]))
        # alpha is (batch*sentence_b) x sentence a
        E4alpha = E4alpha.view(saved_shape)
        alpha = torch.bmm(E4alpha, a)

        # Compare
        ##Concat to each it's weights
        weighted_a = torch.cat((prem_w_e, beta), 2)
        weighted_b = torch.cat((hyp_w_e, alpha), 2)

        ##Feedforward
        v1 = self.G(weighted_a)
        v2 = self.G(weighted_b)

        # Aggregate
        v1 = torch.sum(v1, 1)
        v2 = torch.sum(v2, 1)

        h_in = torch.cat((v1, v2), 1)
        y = self.H(h_in)
        y = self.linear(y)

        if USE_CUDA:
          y = y.to(deviceCPU)

        return y

    def getLabel(self, data):
        _, prediction_argmax = torch.max(data, 1)
        return prediction_argmax

class Run(object):
    def __init__(self, params):
        self.edim = params['EMBEDDING_DIM']
        self.rnn_h_dim = params['RNN_H_DIM']
        self.num_epochs = params['EPOCHS']
        self.batch_size = params['BATCH_SIZE']
        self.train_file = params['TRAIN_FILE']
        self.dev_file = params['DEV_FILE']
        self.test_file = params['TEST_FILE']
        self.test_o_file = params['TEST_O_FILE']
        self.model_file = params['MODEL_FILE']
        self.save_to_file = params['SAVE_TO_FILE']
        self.run_dev = params['RUN_DEV']
        self.learning_rate = params['LEARNING_RATE']
        self.dropout = params['DROPOUT']
        self.acc_data_list = []
        self.load_params = params['LOAD_PARAMS']

    def _save_model_params(self, tagger, wT, lT, optimizer, epoch):
        params = {'tagger' : tagger.state_dict()}
        torch.save(params, self.model_file + "_tagger")
        params = {'wT':wT.saveParams(), 'lT':lT.saveParams()}
        torch.save(params, self.model_file + "_trans")
        params = {'opt':optimizer.state_dict()}
        torch.save(params, self.model_file + "_opt")
        #params = {'epoch':epoch}
        #torch.save(params, self.model_file + "_epoch")

    def _load_opt_params(self, opt):
        params = torch.load(self.model_file + "_opt")
        opt.load_state_dict(params['opt'])

    def _load_translators_params(self, wT, lT):
        params = torch.load(self.model_file + "_trans")
        wT.loadParams(params['wT'])
        lT.loadParams(params['lT'])

    def _load_tagger_params(self, tagger):
        params = torch.load(self.model_file + "_tagger")
        tagger.load_state_dict(params['tagger'])

    def _load_epoch(self):
        params = torch.load(self.model_file + "_epoch")
        return params['epoch']

    def _saveAccData(self, epoch):
        try:
            acc_data = torch.load(FOLDER_PATH + 'accuracy_graphs_data')
        except FileNotFoundError:
            print("No accuracy data file found - creating new")
            acc_data = {}

        acc_data.update({str(epoch): (self.acc_data_list[-1], self.train_accuracy,
                                      self.train_loss)})
        acc_data.update({'epoch':epoch})
        torch.save(acc_data, FOLDER_PATH + 'accuracy_graphs_data')
        params = {'epoch':epoch}
        torch.save(params, self.model_file + "_epoch")

    def _calc_batch_acc(self, tagger, flatten_tag, flatten_label):
        predicted_tags = tagger.getLabel(flatten_tag)
        diff = predicted_tags - flatten_label
        correct_cntr = len(diff[diff == 0])  # tmp
        total_cntr = len(predicted_tags)  # - to_ignore
        return correct_cntr, total_cntr

    def _flat_vecs(self, batch_tag_score, batch_label_list):
        # print(batch_tag_score.shape)
        # print(batch_label_list)
        flatten_tag = batch_tag_score  # .reshape(-1, batch_tag_score.shape[2])
        flatten_label = torch.LongTensor(batch_label_list)  # .reshape(-1))
        # print(flatten_tag)
        # print(flatten_label)
        return flatten_tag, flatten_label

    def runOnDev(self, tagger, padder):
        tagger.eval()
        dev_dataset = As4Dataset(self.dev_file)

        dev_dataset.setTranslators(wT=self.wTran, lT=self.lTran)

        dev_dataloader = DataLoader(dataset=dev_dataset,
                                    batch_size=self.batch_size, shuffle=False,
                                    collate_fn=padder.collate_fn)
        with torch.no_grad():
            correct_cntr = 0
            total_cntr = 0
            for sample in dev_dataloader:
                # batch_data_list, batch_label_list, batch_len_list, padded_sublens = sample
                premise_data, hyp_data, batch_label_list = sample

                # print(premise_data)
                # print(hyp_data)

                batch_tag_score = tagger.forward(premise_data, hyp_data)

                # print(batch_tag_score)

                flatten_tag, flatten_label = self._flat_vecs(batch_tag_score, batch_label_list)

                # calc accuracy
                c, t = self._calc_batch_acc(tagger, flatten_tag, flatten_label)
                correct_cntr += c
                total_cntr += t

        acc = correct_cntr / total_cntr
        self.acc_data_list.append(acc)
        print("Validation accuracy " + str(acc))

        tagger.train()

    def train(self):

        print("Loading data")
        db = DB(self.batch_size);
        train_iter = db.getIter("train")
        print("Done loading data")

        if self.load_params:
          epoch_base = int(self._load_epoch())
        else:
          epoch_base = 0

        print("init tagger")
        tagger = Tagger(embedding_dim=self.edim, projected_dim=self.rnn_h_dim,
                        tagset_size=3, vectors = db.data_field.vocab.vectors, 
                        dropout=self.dropout)

        if self.load_params:
            print("loading model params")
            self._load_tagger_params(tagger)
        print("done")

        if USE_CUDA:
          tagger.to(deviceCuda)

        print("define loss and optimizer")
        loss_function = nn.CrossEntropyLoss()  # ignore_index=len(lTran.tag_dict))
        optimizer = torch.optim.Adagrad(tagger.parameters(), lr=self.learning_rate,
                                        initial_accumulator_value=0.1)  # 0.01)
        if self.load_params:
            self._load_opt_params(optimizer)
        #optimizer = torch.optim.Adadelta(tagger.parameters(), lr=self.learning_rate)
        print("done")

        # print(self.wTran)

        #train_dataloader = DataLoader(dataset=train_dataset,
        #                              batch_size=self.batch_size, shuffle=True,
        #                              collate_fn=padder.collate_fn)

        #print("Starting training")
        #print("data length = " + str(len(train_dataset)))


        if self.run_dev:
            self.runOnDev(tagger, padder)
        for epoch in range(self.num_epochs):
            train_iter.init_epoch()
            loss_acc = 0
            progress1 = 0
            progress2 = 0
            correct_cntr = 0
            total_cntr = 0
            sentences_seen = 0
            for sample in train_iter:
                if progress1 / 100000 > progress2:
                    print("reached " + str(progress2 * 100000))
                    progress2 += 1
                progress1 += self.batch_size
                sentences_seen += self.batch_size

                tagger.zero_grad()

                # batch_data_list, batch_label_list, batch_len_list, padded_sublens = sample
                #premise_data, hyp_data, batch_label_list = sample
                
                premise_data, _ = sample.premise
                hyp_data, _ = sample.hypothesis
                batch_label = (sample.label - torch.ones(sample.label.shape)).long()

                batch_tag_score = tagger.forward(premise_data, hyp_data)
                # flatten_tag, flatten_label = self._flat_vecs(batch_tag_score, batch_label_list)

                # calc accuracy
                # c, t = self._calc_batch_acc(tagger, flatten_tag, flatten_label)
                batch_label_tensor = torch.LongTensor(batch_label)
                c, t = self._calc_batch_acc(tagger, batch_tag_score, batch_label_tensor)
                correct_cntr += c
                total_cntr += t

                # loss = loss_function(flatten_tag, flatten_label)
                loss = loss_function(batch_tag_score, batch_label_tensor)
                loss_acc += loss.item()
                loss.backward()
                optimizer.step()
                
                tagger.zero_grad()

            if self.run_dev:
                self.runOnDev(tagger, padder)
            #print("missed value = " + str(self.wTran.cntr))
            #self.wTran.cntr = 0
            #print("total cntr value = " + str(self.wTran.total_cntr))
            #self.wTran.total_cntr = 0
            print("epoch: " + str(epoch) + " " + str(loss_acc))
            self.train_accuracy = correct_cntr/total_cntr
            self.train_loss = loss_acc
            print("Train accuracy " + str(correct_cntr/total_cntr))

            if self.save_to_file:
              if (epoch % 5) == 4:
                print("saving model params")
                self._save_model_params(tagger, self.wTran, self.lTran, optimizer,
                                        epoch_base + epoch)
                print("done saving model params")

            if self.run_dev:
              self._saveAccData(epoch_base + epoch)
        # if (sys.argv[1] == 'save') or (sys.argv[1] == 'loadsave'):
        # self._save_model_params(tagger, self.wTran, self.lTran)
        # torch.save(tagger.state_dict(), 'bilstm_params.pt')


FAVORITE_RUN_PARAMS = {
    'EMBEDDING_DIM': 300,
    'RNN_H_DIM': 200,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 0.05
}

if __name__ == "__main__":
    #FOLDER_PATH = "./data/snli_1.0/"
    train_file = FOLDER_PATH + "snli_1.0_train.jsonl"
                     #"sys.argv[1]
    model_file = FOLDER_PATH + '32Bbatch' #sys.argv[2]
    epochs = 50 #int(sys.argv[3])
    run_dev = False #sys.argv[4]
    dev_file = FOLDER_PATH + "snli_1.0_dev.jsonl"

    RUN_PARAMS = FAVORITE_RUN_PARAMS
    RUN_PARAMS.update({
                'TRAIN_FILE': train_file,
                'DEV_FILE' : dev_file,
                'TEST_FILE': None, #test_file,
                'TEST_O_FILE': None, #test_o_file,
                'MODEL_FILE': model_file,
                'SAVE_TO_FILE': False,
                'RUN_DEV' : run_dev,
                'EPOCHS' : epochs,
                'LOAD_PARAMS': False,
                'DROPOUT' : True})

    run = Run(RUN_PARAMS)

    run.train()
