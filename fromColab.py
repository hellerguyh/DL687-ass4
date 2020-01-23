import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random as r
#from torchnlp.word_to_vector import GloVe
import json
from spacy.lang.en import English

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
#from torchnlp.word_to_vector import GloVe

import spacy

spacy.load('en')







import sys

FOLDER_PATH = "./gdrive/My Drive/Master/DL687/As4/1606.01933/data/snli_1.0/"
DEBUG = True


def DEBUG_PRINT(x):
    if DEBUG:
        print(x)

deviceCuda = torch.device("cuda")
deviceCPU = torch.device("cpu")
USE_CUDA = False
USE_840 = False #True
RUNNING_LOCAL = True
if RUNNING_LOCAL:
    FOLDER_PATH = './data/snli_1.0/'
    USE_CUDA = False



#if torch.cuda.is_available():
if USE_CUDA:
    torch.cuda.set_device(0)
    device = torch.device('cuda:{}'.format(0))
else:
    device = torch.device('cpu')


# set up fields
TEXT = data.Field(lower=True, batch_first=True, tokenize='spacy', tokenizer_language='en') #include_lengths=True, tokenize='spacy', tokenizer_language='en')
LABEL = data.Field(sequential=False)

# make splits for data
train, dev, test = datasets.SNLI.splits(TEXT, LABEL)

# build the vocabulary
if USE_840:
  TEXT.build_vocab(train, vectors=GloVe(name='840B', dim=100, cache="./glove_cache/"))
else:
  TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=100, cache="./glove_cache/"))

LABEL.build_vocab(train)

# make iterator for splits
train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=4, device=device)


class BatchWrapper:
    def __init__(self, dl, x1_var, x2_var, y_vars):
        self.dl, self.x1_var, self.x2_var, self.y_vars = dl, x1_var, x2_var, y_vars  # we pass in the list of attributes for x

    def __iter__(self):
        for batch in self.dl:
            x1 = getattr(batch, self.x1_var)
            x2 = getattr(batch, self.x2_var)

            if self.y_vars is not None:  # we will concatenate y into a single tensor
                #print(getattr(batch, self.y_vars))
                y = torch.Tensor(getattr(batch, self.y_vars).float())#y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x1, x2, y)

    def __len__(self):
        return len(self.dl)


train_dl = BatchWrapper(train_iter, "premise", "hypothesis", "label")
dev_dl = BatchWrapper(dev_iter, "premise", "hypothesis", "label")
test_dl = BatchWrapper(test_iter, "premise", "hypothesis", "label")


class Tagger(nn.Module):
    def __init__(self, embedding_dim, projected_dim, tagset_size,
                 translator, f_dim=200, v_dim=200, dropout=False):
        super(Tagger, self).__init__()
        self.embedding_dim = embedding_dim

        # # Creat Embeddings
        # vecs = GLOVE_DATA.vectors
        # vecs = vecs/torch.norm(vecs, dim=1, keepdim=True)
        # ## Add to glove vectors 2 vectors for unknown and padding:
        # for i in range(100):
        #     #pad = torch.rand((1, vecs[0].shape[0]))
        #     pad = torch.normal(mean=torch.zeros(1, vecs[0].shape[0]), std=1)
        #     vecs = torch.cat((vecs, pad), 0)
        # pad = torch.zeros((1, vecs[0].shape[0]))
        # vecs = torch.cat((vecs, pad), 0)

        vocab = TEXT.vocab
        self.wembeddings = nn.Embedding(len(vocab), self.embedding_dim)
        self.wembeddings.weight.data.copy_(vocab.vectors)

        # self.wembeddings = nn.Embedding.from_pretrained(embeddings=vecs, freeze=True,
        #                                                 padding_idx=translator.getPaddingIndex()['w'])
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
        premise_data, hyp_data

        padded_premise_w = premise_data#['w_data']
        #premise_w_lens = premise_data['w_lens']

        padded_hyp_w = hyp_data#4['w_data']
        #hyp_w_lens = hyp_data['w_lens']

        batch_size = len(padded_premise_w)

        if USE_CUDA:
          padded_premise_w = torch.from_numpy(padded_premise_w).to(deviceCuda)
          padded_hyp_w = torch.from_numpy(padded_hyp_w).to(deviceCuda)

        prem_w_e = self.wembeddings(torch.tensor(padded_premise_w).long())
        hyp_w_e = self.wembeddings(torch.tensor(padded_hyp_w).long())

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



import tqdm

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

    def _save_model_params(self, tagger, wT, lT, optimizer):
        try:
            params = torch.load(self.model_file)
        except FileNotFoundError:
            print("No model params file found - creating new model params")
            params = {}

        flavor_params = {}
        flavor_params.update({'tagger' : tagger.state_dict()})
        flavor_params.update({'wT' : wT.saveParams()})
        flavor_params.update({'lT' : lT.saveParams()})
        flavor_params.update({'optimizer': optimizer.state_dict()})
        params.update({'model_params' : flavor_params})
        torch.save(params, self.model_file)

    def _load_opt_params(self, opt):
        params = torch.load(self.model_file)
        flavor_params = params['model_params']
        opt.load_state_dict(flavor_params['optimizer'])

    def _load_translators_params(self, wT, lT):
        params = torch.load(self.model_file)
        flavor_params = params['model_params']
        wT.loadParams(flavor_params['wT'])
        lT.loadParams(flavor_params['lT'])

    def _load_tagger_params(self, tagger):
        params = torch.load(self.model_file)
        flavor_params = params['model_params']
        tagger.load_state_dict(flavor_params['tagger'])

    def _saveAccData(self):
        try:
            acc_data = torch.load(FOLDER_PATH + 'accuracy_graphs_data')
        except FileNotFoundError:
            print("No accuracy data file found - creating new")
            acc_data = {}

        acc_data.update({'accuracy': self.acc_data_list})
        torch.save(acc_data, FOLDER_PATH + 'accuracy_graphs_data')

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
        flatten_label = batch_label_list.long() #torch.LongTensor(batch_label_list)  # .reshape(-1))
        # print(flatten_tag)
        # print(flatten_label)
        return flatten_tag, flatten_label

    def runOnDev(self, tagger):
        tagger.eval()
        with torch.no_grad():
            correct_cntr = 0
            total_cntr = 0
            for sample in tqdm.tqdm(dev_dl):
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

        print("Done loading data")




        print("init tagger")
        tagger = Tagger(embedding_dim=self.edim, projected_dim=self.rnn_h_dim,
                        translator=None, tagset_size=len(LABEL.vocab),  # + 1,
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



        if self.run_dev:
            self.runOnDev(tagger)
        for epoch in range(self.num_epochs):
            loss_acc = 0
            progress1 = 0
            progress2 = 0
            correct_cntr = 0
            total_cntr = 0
            sentences_seen = 0
            for sample in tqdm.tqdm(train_dl):
                if progress1 / 100000 > progress2:
                    print("reached " + str(progress2 * 100000))
                    progress2 += 1
                progress1 += self.batch_size
                sentences_seen += self.batch_size

                tagger.zero_grad()

                # batch_data_list, batch_label_list, batch_len_list, padded_sublens = sample
                premise_data, hyp_data, batch_label_list = sample
                batch_tag_score = tagger.forward(premise_data, hyp_data)

                #flatten_tag, flatten_label = self._flat_vecs(batch_tag_score, batch_label_list)

                # calc accuracy
                # c, t = self._calc_batch_acc(tagger, flatten_tag, flatten_label)
                batch_label_tensor = batch_label_list.long() #torch.LongTensor(batch_label_list)
                c, t = self._calc_batch_acc(tagger, batch_tag_score, batch_label_tensor)
                correct_cntr += c
                total_cntr += t

                # print(flatten_tag)
                # print(flatten_label)
                #loss = loss_function(flatten_tag, flatten_label)

                loss = loss_function(batch_tag_score, batch_label_tensor)
                loss_acc += loss.item()
                loss.backward()
                optimizer.step()

            if self.run_dev:
                self.runOnDev(tagger)

            print("epoch: " + str(epoch) + " " + str(loss_acc))
            print("Train accuracy " + str(correct_cntr/total_cntr))

            if self.save_to_file:
                print("saving model params")
                self._save_model_params(tagger, self.wTran, self.lTran, optimizer)

        if self.run_dev:
            self._saveAccData()
        # if (sys.argv[1] == 'save') or (sys.argv[1] == 'loadsave'):
        # self._save_model_params(tagger, self.wTran, self.lTran)
        # torch.save(tagger.state_dict(), 'bilstm_params.pt')


FAVORITE_RUN_PARAMS = {
    'EMBEDDING_DIM': 100,
    'RNN_H_DIM': 200,
    'BATCH_SIZE': 100,
    'LEARNING_RATE': 0.05
}

if __name__ == "__main__":
    #FOLDER_PATH = "./data/snli_1.0/"
    train_file = FOLDER_PATH + "small_dataset.jsonl" #snli_1.0_train.jsonl"
                     #"sys.argv[1]
    model_file = FOLDER_PATH + 'SOMEMODEL' #sys.argv[2]
    epochs = 100 #int(sys.argv[3])
    run_dev = True #sys.argv[4]
    dev_file = FOLDER_PATH + "snli_1.0_dev.jsonl"

    RUN_PARAMS = FAVORITE_RUN_PARAMS
    RUN_PARAMS.update({
                'TRAIN_FILE': train_file,
                'DEV_FILE' : dev_file,
                'TEST_FILE': None, #test_file,
                'TEST_O_FILE': None, #test_o_file,
                'MODEL_FILE': model_file,
                'SAVE_TO_FILE': False, #True,
                'RUN_DEV' : run_dev,
                'EPOCHS' : epochs,
                'LOAD_PARAMS': False,#True,
                'DROPOUT' : True})

    run = Run(RUN_PARAMS)

    run.train()
