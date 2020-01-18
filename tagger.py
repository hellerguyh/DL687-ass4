import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random as r

from torchnlp.word_to_vector import GloVe
import json

import sys

FOLDER_PATH = None
DEBUG = True


def DEBUG_PRINT(x):
    if DEBUG:
        print(x)


GLOVE_DATA = GloVe(name='6B', dim=300)


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


''' Seems like gloves works on words! not indexes!!!! '''


class As4Dataset(Dataset):
    def __init__(self, file_path, is_test_data=False, is_train_data=True):

        self.file_path = file_path

        dataset = []
        sample_w = []
        sample_t = []
        word_list = []
        tag_list = []

        with open(file_path, "r") as df:
            for line in df:
                line = line.strip()
                line = json.loads(line)
                if (line['gold_label'] == 'entailment') or (line['gold_label'] == 'contradiction') or (
                        line['gold_label'] == 'neutral'):
                    dataset.append({
                        'premise': line['sentence1'].split(),
                        'hypothesis': line['sentence2'].split(),
                        'label': line['gold_label']
                    })

        # self.word_set = set(word_list)
        # self.tag_set = set(tag_list)
        self.dataset = dataset
        self.is_test_data = is_test_data
        self.is_train_data = is_train_data

    def __len__(self):
        return len(self.dataset)

    def setTranslators(self, wT, lT):
        self.wT = wT
        self.lT = lT

    def toIndexes(self, wT, lT):
        self.dataset = [{'premise': wT.translate(data['premise']), 'hypothesis': wT.translate(data['hypothesis']),
                         'label': lT.translate(data['label'])} for data in self.dataset]
        # self.dataset = [(wT.translate(data[0], self.is_train_data), lT.translate(data[1]) if self.is_test_data==False else None, data[2]) for data in self.dataset]

    def __getitem__(self, index):
        data = self.dataset[index]
        return {'premise': self.wT.translate(data['premise']), 'hypothesis': self.wT.translate(data['hypothesis']),
                'label': self.lT.translate(data['label'])}
        # return self.dataset[index]


'''
Translates word represented as strings to indexes at the embedding table
should be init=True at train dataset and init=False on test/dev dataset
'''


class WTranslator(object):
    def __init__(self, init=True):
        if init:
            self.wdict = GLOVE_DATA.token_to_index
            unknown_idx = len(GLOVE_DATA)
            self.wdict.update({"UNKNOWN": unknown_idx})
            self.wpadding_idx = unknown_idx + 1

    def getPaddingIndex(self):
        return {'w': self.wpadding_idx}

    def saveParams(self):
        return {'wdict': self.wdict}

    def loadParams(self, params):
        self.wdict = params['wdict']

    def _dictHandleExp(self, dic, val):
        try:
            return dic[val]
        except KeyError:
            return dic['UNKNOWN']

    def _translate1(self, word_list):
        # Note that GLOVE is using only lower case words, hence we need to lower case the words
        return [self._dictHandleExp(self.wdict, word.lower()) for word in word_list]

    def translate(self, word_list):
        first = np.array(self._translate1(word_list))
        return {'word': first}

    def getLengths(self):
        return {'word': len(self.wdict)}


class TagTranslator(object):
    def __init__(self, init=True):
        if init:
            tagset = ['entailment', 'contradiction', 'neutral']
            self.tag_dict = list2dict(tagset)

    def translate(self, tag):
        return self.tag_dict[tag]

    def getLengths(self):
        return {'tag': len(self.tag_dict)}

    def getPaddingIdx(self):
        return {'tag': len(self.tag_dict)}

    def saveParams(self):
        return {'tag': self.tag_dict}

    def loadParams(self, params):
        self.tag_dict = params['tag']


''' Padds a batch of examples so all will be at the same length 
    Returns also the original lengths of the hypothesis and premise 

    Note: data is padded with "pad_index" - an index cordinated with the
    embedding layer to be a padding index and hence output zeros.
'''


class Padding(object):
    def __init__(self, wT, lT):
        self.wT = wT
        self.lT = lT
        self.wPadIndex = self.wT.getPaddingIndex()['w']

    def padData(self, data_b, len_b, max_l, padIndex):
        batch_size = len(len_b)
        padded_data = np.ones((batch_size, max_l)) * padIndex
        for i, data in enumerate(data_b):
            padded_data[i][:len_b[i]] = data  # first embeddings
        return padded_data

    def collate_fn(self, data):
        # data.sort(key=lambda x: x[2], reverse=True)

        tag_b = [d['label'] for d in data]

        premise_w_lens = [len(d['premise']['word']) for d in data]
        data_premise = [d['premise']['word'] for d in data]
        padded_premise_w = self.padData(data_premise, premise_w_lens, max(premise_w_lens), self.wPadIndex)

        hyp_w_lens = [len(d['hypothesis']['word']) for d in data]
        data_hyp = [d['hypothesis']['word'] for d in data]
        padded_hyp_w = self.padData(data_hyp, hyp_w_lens, max(hyp_w_lens), self.wPadIndex)

        premise_data = {'w_data': padded_premise_w, 'w_lens': premise_w_lens}
        hyp_data = {'w_data': padded_hyp_w, 'w_lens': hyp_w_lens}
        return premise_data, hyp_data, tag_b


class MLP(nn.Module):
    def __init__(self, i_dim, o_dim, dropout=0):
        super(MLP, self).__init__()
        use_dropout = dropout > 0
        self.layers = []

        self.layers.append(nn.Linear(i_dim, o_dim))
        if use_dropout:
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(o_dim, o_dim))
        if use_dropout:
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.ReLU())

    def forward(self, vec):
        r = vec
        for layer in self.layers:
            r = layer(r)
        return r


class Attention(nn.Module):
    def __init__(self, hidden_dim, f_dim):
        super(Attention, self).__init__()
        self.F = MLP(hidden_dim, f_dim, 0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, a, b):
        # a is of shape batch x sentence_a x hidden_dim
        # b is of shape batch x sentence_b x hidden_dim
        batch_size = a.shape[0]
        fa = self.F(a)
        fb = self.F(b)

        # We want to calculate e_ij = fa_i * fb_j
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

        return beta, alpha


class Tagger(nn.Module):
    def __init__(self, embedding_dim, projected_dim, tagset_size,
                 translator, f_dim=200, v_dim=200, dropout=False):
        super(Tagger, self).__init__()
        self.embedding_dim = embedding_dim

        # Creat Embeddings
        vecs = GLOVE_DATA.vectors
        ## Add to glove vectors 2 vectors for unknown and padding:
        pad = torch.zeros((2, vecs[0].shape[0]))
        vecs = torch.cat((vecs, pad), 0)
        self.wembeddings = nn.Embedding.from_pretrained(embeddings=vecs, freeze=True,
                                                        padding_idx=translator.getPaddingIndex()['w'])
        ## project down the vectors to 200dim
        self.project = nn.Linear(embedding_dim, projected_dim)
        self.attention = Attention(hidden_dim=projected_dim, f_dim=f_dim)
        self.G = MLP(f_dim * 2, v_dim, 0.2)
        self.H = MLP(v_dim * 2, v_dim, 0.2)
        self.linear = nn.Linear(v_dim, tagset_size)

    def forward(self, premise_data, hyp_data):
        premise_data, hyp_data

        padded_premise_w = premise_data['w_data']
        premise_w_lens = premise_data['w_lens']

        padded_hyp_w = hyp_data['w_data']
        hyp_w_lens = hyp_data['w_lens']

        batch_size = len(padded_premise_w)

        prem_w_e = self.wembeddings(torch.tensor(padded_premise_w).long())
        hyp_w_e = self.wembeddings(torch.tensor(padded_hyp_w).long())

        # Project the embeddings to smaller vector
        prem_w_e = self.project(prem_w_e)
        hyp_w_e = self.project(hyp_w_e)

        beta, alpha = self.attention(prem_w_e, hyp_w_e)

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
        # dev_dataset.toIndexes(wT = self.wTran, lT = self.lTran)

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
        train_dataset = As4Dataset(self.train_file, is_test_data=False, is_train_data=True)
        print("Done loading data")

        self.wTran = WTranslator()
        self.lTran = TagTranslator()

        print("translate to indexes")
        # train_dataset.toIndexes(wT = self.wTran, lT = self.lTran)
        train_dataset.setTranslators(wT=self.wTran, lT=self.lTran)
        print("done")

        print("init tagger")
        tagger = Tagger(embedding_dim=self.edim, projected_dim=self.rnn_h_dim,
                        translator=self.wTran, tagset_size=self.lTran.getLengths()['tag'],  # + 1,
                        dropout=self.dropout)
        print("done")

        print("define loss and optimizer")
        loss_function = nn.CrossEntropyLoss()  # ignore_index=len(lTran.tag_dict))
        optimizer = torch.optim.Adam(tagger.parameters(), lr=self.learning_rate)  # 0.01)
        print("done")

        print("init padder")
        padder = Padding(self.wTran, self.lTran)
        print("done")

        # print(self.wTran)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size, shuffle=True,
                                      collate_fn=padder.collate_fn)

        print("Starting training")
        print("data length = " + str(len(train_dataset)))

        if self.run_dev:
            self.runOnDev(tagger, padder)
        for epoch in range(self.num_epochs):
            loss_acc = 0
            progress1 = 0
            progress2 = 0
            correct_cntr = 0
            total_cntr = 0
            sentences_seen = 0
            for sample in train_dataloader:
                if progress1 / 1000 == progress2:
                    print("reached " + str(progress2 * 1000))
                    progress2 += 1
                progress1 += self.batch_size
                sentences_seen += self.batch_size

                tagger.zero_grad()
                # batch_data_list, batch_label_list, batch_len_list, padded_sublens = sample
                premise_data, hyp_data, batch_label_list = sample

                batch_tag_score = tagger.forward(premise_data, hyp_data)

                # flatten_tag, flatten_label = self._flat_vecs(batch_tag_score, batch_label_list)

                # calc accuracy
                # c, t = self._calc_batch_acc(tagger, flatten_tag, flatten_label)
                batch_label_tensor = torch.LongTensor(batch_label_list)
                c, t = self._calc_batch_acc(tagger, batch_tag_score, batch_label_tensor)
                correct_cntr += c
                total_cntr += t

                # loss = loss_function(flatten_tag, flatten_label)
                loss = loss_function(batch_tag_score, batch_label_tensor)
                loss_acc += loss.item()
                loss.backward()
                optimizer.step()

            self.runOnDev(tagger, padder)
            print("epoch: " + str(epoch) + " " + str(loss_acc))
            print("Train accuracy " + str(correct_cntr/total_cntr))

        if self.save_to_file:
            self._save_model_params(tagger, self.wTran, self.lTran)

        if self.run_dev:
            self._saveAccData()
        # if (sys.argv[1] == 'save') or (sys.argv[1] == 'loadsave'):
        # self._save_model_params(tagger, self.wTran, self.lTran)
        # torch.save(tagger.state_dict(), 'bilstm_params.pt')


FAVORITE_RUN_PARAMS = {
    'EMBEDDING_DIM': 300,
    'RNN_H_DIM': 200,
    'EPOCHS': 20,
    'BATCH_SIZE': 4,
    'LEARNING_RATE': 0.05
}

if __name__ == "__main__":
    FOLDER_PATH = "./"
    train_file = FOLDER_PATH + "snli_1.0/snli_1.0_short.0/snli_1.0_train.jsonl"
                     #"sys.argv[1]
    model_file = 'SOMEMODEL' #sys.argv[2]
    epochs = 100 #int(sys.argv[3])
    run_dev = True #sys.argv[4]
    dev_file = FOLDER_PATH + "snli_1.0/snli_1.0_short.0/snli_1.0_dev.jsonl"

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
                'DROPOUT' : True})

    run = Run(RUN_PARAMS)

    run.train()
