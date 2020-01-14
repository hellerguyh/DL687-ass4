import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random as r

from torchnlp.word_to_vector import GloVe
import json

import sys

FOLDER_PATH =None
DEBUG = True
def DEBUG_PRINT(x):
  if DEBUG:
    print(x)


GLOVE_DATA = GloVe(name='6B', dim=100)

def list2dict(lst):
    it = iter(lst)
    indexes = range(len(lst))
    res_dct = dict(zip(it, indexes))
    return res_dct

def reverseDict(d):
    vals = ['']*len(d.keys())
    for k in d.keys():
        vals[d[k]] = k
    return vals

''' Seems like gloves works on words! not indexes!!!! '''

class As3Dataset(Dataset):
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
                if (line['gold_label'] == 'entailment') or (line['gold_label'] == 'contradiction') or (line['gold_label'] == 'neutral'): 
                    dataset.append({
                                    'premise': line['sentence1'].split(),
                                    'hypothesis':line['sentence2'].split(),
                                    'label':line['gold_label']
                                    })
        
        #self.word_set = set(word_list)
        #self.tag_set = set(tag_list)
        self.dataset = dataset
        self.is_test_data = is_test_data
        self.is_train_data = is_train_data

    def __len__(self):
        return len(self.dataset)

    def setTranslators(self, wT, lT):
        self.wT = wT
        self.lT = lT

    def toIndexes(self, wT, lT):
        self.dataset = [{'premise':wT.translate(data['premise']), 'hypothesis':wT.translate(data['hypothesis']), 'label':lT.translate(data['label'])} for data in self.dataset]
        #self.dataset = [(wT.translate(data[0], self.is_train_data), lT.translate(data[1]) if self.is_test_data==False else None, data[2]) for data in self.dataset]

    def __getitem__(self, index):
        data = self.dataset[index]
        return {'premise': self.wT.translate(data['premise']), 'hypothesis': self.wT.translate(data['hypothesis']), 'label': self.lT.translate(data['label'])}
        #return self.dataset[index]

class WTranslator(object):
    def __init__(self, init=True):
        if init:
            self.wdict = GLOVE_DATA.token_to_index
            unknown_idx = len(GLOVE_DATA)
            self.wdict.update({"UNKNOWN":unknown_idx})
            self.wpadding_idx = unknown_idx + 1
            
            #wordset.update(["UNKNOWN"])
            cset = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
            cset.append("UNKNOWN")
            self.cdict = list2dict(cset)
            self.cpadding_idx = len(self.cdict)

    def getPaddingIndex(self):
        return {'w':self.wpadding_idx, 'c':self.cpadding_idx}

    def saveParams(self):
        return {'cdict':self.cdict, 'wdict':self.wdict}

    def loadParams(self, params):
        self.cdict = params['cdict']
        self.wdict = params['wdict']

    def _dictHandleExp(self, dic, val):
      try: 
        return dic[val]
      except KeyError:
        return dic['UNKNOWN']
    
    def _translate1(self, word_list):
        # Note that GLOVE is using only lower case words, hence we need to lower case the words
        return [self._dictHandleExp(self.wdict, word.lower()) for word in word_list]

    def _translate2(self, word_list):
        letter_trans = [np.array([self._dictHandleExp(self.cdict, l) for l in word]) for word in word_list]
        lengths = [len(word) for word in word_list]
        return [letter_trans, lengths]

    def translate(self, word_list):
        first = np.array(self._translate1(word_list))
        second = self._translate2(word_list)
        return {'word': first, 'chars': second}

    def getLengths(self):
        return {'word' : len(self.wdict), 'c' : len(self.cdict)}

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
        return {'tag':self.tag_dict}

    def loadParams(self, params):
        self.tag_dict = params['tag']


class MyEmbedding(nn.Module):
    def __init__(self, embedding_dim, translator, c_embedding_dim):
        super(MyEmbedding, self).__init__()
        padding_idx = translator.getPaddingIndex()['w']
        num_embedding = translator.getLengths()['word']
        #self.wembeddings = nn.Embedding(num_embeddings = num_embedding + 1, embedding_dim = embedding_dim, padding_idx = padding_idx)
        vecs = GLOVE_DATA.vectors
        pad = torch.zeros((2,vecs[0].shape[0]))
        vecs = torch.cat((vecs, pad), 0)
        self.wembeddings = nn.Embedding.from_pretrained(embeddings = vecs, freeze=False,
                                                        padding_idx = padding_idx) 

        padding_idx = translator.getPaddingIndex()['c']
        num_embedding = translator.getLengths()['c']
        self.cembeddings = nn.Embedding(num_embeddings = num_embedding + 1, embedding_dim = c_embedding_dim, padding_idx = padding_idx)
    
    def forward(self, data):
        word_embeds = self.wembeddings(torch.tensor(data[0]).long())
        char_embeds = self.cembeddings(torch.tensor(data[1]).long())
        return (word_embeds, char_embeds)

class Padding(object):
    def __init__(self, wT, lT):
        self.wT = wT
        self.lT = lT
        
        self.wPadIndex = self.wT.getPaddingIndex()['w']
        self.cPadIndex = self.wT.getPaddingIndex()['c']
        
    def padData(self, data_b, len_b, max_l, padIndex):
        batch_size = len(len_b)
        padded_data = np.ones((batch_size, max_l))*padIndex
        for i, data in enumerate(data_b):
            padded_data[i][:len_b[i]] = data #first embeddings
        return padded_data

    def padTag(self, tag_b, len_b, max_l, padIndex):
        batch_size = len(len_b)
        padded_tag = np.ones((batch_size, max_l))*padIndex
        for i,tag in enumerate(tag_b):
            padded_tag[i][:len_b[i]] = np.array(tag)
        return padded_tag
  
    def padList(self, data_b, lens_b,  max_l):
        # Expect data_b shape = <batch_size>, <sentence_len>, [<word_len>, 1]
        # returns: <batch_size>, <max sentence len>, <max word_len>

        w_max_l = 0
        for batch in data_b:
            sentence, word_len = batch
            m = max(word_len)
            if m > w_max_l:
                w_max_l = m

        batch_size = len(lens_b)
        padded_words = np.ones((batch_size, max_l, w_max_l))*self.cPadIndex
        padded_lens = np.ones((batch_size, max_l))
        for i, batch in enumerate(data_b):
            sentence, words_len = batch
            for j, word in enumerate(sentence):
                word_len = words_len[j]
                padded_words[i][j][:word_len] = word
                padded_lens[i][j] = word_len
        
        return padded_words, padded_lens

    def collate_fn(self, data):
        #data.sort(key=lambda x: x[2], reverse=True)

        tag_b = [d['label'] for d in data]
        
        premise_w_lens = [len(d['premise']['word']) for d in data]
        data_premise = [d['premise']['word'] for d in data]
        padded_premise_w = self.padData(data_premise, premise_w_lens, max(premise_w_lens), self.wPadIndex)
        
        hyp_w_lens = [len(d['hypothesis']['word']) for d in data]
        data_hyp = [d['hypothesis']['word'] for d in data]
        padded_hyp_w = self.padData(data_hyp, hyp_w_lens, max(hyp_w_lens), self.wPadIndex)

        data_premise = [d['premise']['chars'] for d in data]
        padded_premise_c, padded_premise_sublens = self.padList(data_premise, premise_w_lens, max(premise_w_lens))

        data_hyp = [d['hypothesis']['chars'] for d in data]
        padded_hyp_c, padded_hyp_sublens = self.padList(data_hyp, hyp_w_lens, max(hyp_w_lens))

        premise_data = {'w_data': padded_premise_w, 'w_lens': premise_w_lens, 'c_data': padded_premise_c, 'c_lens': padded_premise_sublens}
        hyp_data = {'w_data': padded_hyp_w, 'w_lens': hyp_w_lens, 'c_data': padded_hyp_c, 'c_lens': padded_hyp_sublens}
        return premise_data, hyp_data, tag_b

    

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_rnn_dim, tagset_size,
                translator, c_embedding_dim, filters_dim = 100, filters = [1,3,5], dropout=False, add_tanh=False,
                num_lstm_layers = 3):
        super(BiLSTM, self).__init__()
        self.c_embeds_dim = c_embedding_dim
        self.embedding_dim = embedding_dim
        # Embedding layers
        padding_idx = translator.getPaddingIndex()['w']
        num_embedding = translator.getLengths()['word']
        vecs = GLOVE_DATA.vectors
        pad = torch.zeros((2,vecs[0].shape[0]))
        vecs = torch.cat((vecs, pad), 0)
        self.wembeddings = nn.Embedding.from_pretrained(embeddings = vecs, freeze=False,
                                                        padding_idx = padding_idx) 

        padding_idx = translator.getPaddingIndex()['c']
        num_embedding = translator.getLengths()['c']
        self.cembeddings = nn.Embedding(num_embeddings = num_embedding + 1, embedding_dim = c_embedding_dim, padding_idx = padding_idx)

        self.dropout_0 = nn.Dropout()  
        #self.lstmc = nn.LSTM(input_size = c_embedding_dim, hidden_size = embedding_dim,
        #                    batch_first = True)
        self.conv_list = []
        for kernel in filters:
            self.conv_list.append(nn.Conv2d(c_embedding_dim, filters_dim, kernel))
        self.lstm_list = []
        for i in range(num_lstm_layers):
            self.lstm_list.append(nn.LSTM(input_size = embedding_dim*(len(filters) + 1), hidden_size = hidden_rnn_dim,
                            bidirectional=True, num_layers=1, batch_first=True))
        '''
        To be replaced 
        self.linear1 = nn.Linear(hidden_rnn_dim*2, tagset_size)
        self.dropout_1 = nn.Dropout()  
        self.lineare = nn.Linear(embedding_dim*2, embedding_dim)
        self.dropout_e = nn.Dropout()
        self.dropout = dropout
        self.add_tanh = add_tanh
        '''

    def conv(self, e_batch):
        for conv_layer in self.conv_list:
            c = conv_layer(e_batch)
            c = nn.functional.relu(c)
            c = torch.max(c,2)
        return c

    def forward(self, sample):
        premise_data, hyp_data, tag_b = sample

        padded_premise_w = premise_data['w_data']
        premise_w_lens = premise_data['w_lens']
        padded_premise_c = premise_data['c_data']

        padded_hyp_w = hyp_data['w_data']
        hyp_w_lens = hyp_data['w_lens']
        padded_hyp_c = hyp_data['c_data']

        batch_size = len(padded_premise_w)

        prem_w_e = self.wembeddings(torch.tensor(padded_premise_w).long())
        hyp_w_e = self.wembeddings(torch.tensor(padded_hyp_w).long())
        prem_c_e = self.cembeddings(torch.tensor(padded_premise_c).long())
        hyp_c_e = self.cembeddings(torch.tensor(padded_hyp_c).long())
       
        print(prem_c_e.shape)
        ''' Currently it's NumWordsXMaxNumCharsXEmbeddingSize -> need to rehsape into 
        EmbeddingsSizeXNumWordsXMaxNumChars because the embedding size is the channels '''
        prem_c_conv = self.conv(prem_c_e)
        print(prem_c_conv) 
        #embeds_list = self.embeddings.forward(data_list)
        
        embeds_word = embeds_list[0]
        embeds_char = embeds_list[1]
        char_data_list = data_list[1]

        #lstm_embeds_word = self.runLSTMc(char_data_list, embeds_char, padded_sublens) 

        e_joined = torch.cat((embeds_word, lstm_embeds_word), dim=2)
        flatten = e_joined.reshape(-1, e_joined.shape[2])
        if self.dropout:
            e_joined = self.dropout_e(e_joined)
        le_out = self.lineare(e_joined)
        if self.add_tanh:
            le_out = torch.tanh(le_out)
        embeds_out = le_out.reshape(batch_size, e_joined.shape[1], self.embedding_dim)
        
        if self.dropout:
            embeds_out = self.dropout_0(embeds_out)

        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds_out, len_list, batch_first=True)
        lstm_out, _ = self.lstm(packed_embeds)
        unpacked_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first = True)

        flatten = unpacked_lstm_out.reshape(-1, unpacked_lstm_out.shape[2])
        if self.dropout:
            flatten = self.dropout_1(flatten)
        o_ln1 = self.linear1(flatten)
        shaped = o_ln1.reshape(batch_size, unpacked_lstm_out.shape[1], o_ln1.shape[1])
        return shaped

    def getLabel(self, data):
        _, prediction_argmax = torch.max(data, 1)
        return prediction_argmax


class Run(object):
    def __init__(self, params):
        self.edim = params['EMBEDDING_DIM']
        self.rnn_h_dim = params['RNN_H_DIM']
        self.num_epochs = params['EPOCHS']
        self.batch_size = params['BATCH_SIZE']
        self.c_embedding_dim = params['CHAR_EMBEDDING_DIM']
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

    def _save_model_params(self, tagger, wT, lT):
        try:
            params = torch.load(self.model_file)
        except FileNotFoundError:
            print("No model params file found - creating new model params")
            params = {}

        flavor_params = {}
        flavor_params.update({'tagger' : tagger.state_dict()})
        flavor_params.update({'wT' : wT.saveParams()})
        flavor_params.update({'lT' : lT.saveParams()})
        params.update({'ModelParams' : flavor_params})
        torch.save(params, self.model_file)

    def _load_translators_params(self, wT, lT):
        params = torch.load(self.model_file)
        flavor_params = params['ModelParams']
        wT.loadParams(flavor_params['wT'])
        lT.loadParams(flavor_params['lT'])

    def _load_bilstm_params(self, tagger):
        params = torch.load(self.model_file)
        flavor_params = params[str('ModelParams')]
        tagger.load_state_dict(flavor_params['tagger'])

    def _calc_batch_acc(self, tagger, flatten_tag, flatten_label): 
        predicted_tags = tagger.getLabel(flatten_tag)
        diff = predicted_tags - flatten_label
        no_diff = (diff == 0)
        padding_mask = (flatten_label == self.lTran.getLengths()['tag'])
        if self.ignore_Os:
            Os_mask = (flatten_label == self.lTran.tag_dict['O'])
            no_diff_and_padding_label = no_diff*(padding_mask + Os_mask)
            no_diff_and_padding_label = (no_diff_and_padding_label > 0)
        else:
            no_diff_and_padding_label = no_diff*padding_mask

        to_ignore = len(no_diff_and_padding_label[no_diff_and_padding_label == True])
        tmp = len(diff[diff == 0]) - to_ignore
        if tmp < 0:
            raise Exception("non valid tmp value")
        correct_cntr = tmp 
        total_cntr = len(predicted_tags) - to_ignore
        return correct_cntr, total_cntr

    def _flat_vecs(self, batch_tag_score, batch_label_list):
        flatten_tag = batch_tag_score.reshape(-1, batch_tag_score.shape[2])
        flatten_label = torch.LongTensor(batch_label_list.reshape(-1))
        return flatten_tag, flatten_label

    def runOnDev(self, tagger, padder):
        tagger.eval()
        dev_dataset = As3Dataset(self.dev_file, False, False)
        #dev_dataset.toIndexes(wT = self.wTran, lT = self.lTran)
        dev_dataset.setTranslators(wT = self.wTran, lT = self.lTran)
        dev_dataloader = DataLoader(dataset=dev_dataset,
                                    batch_size=self.batch_size, shuffle=False,
                                    collate_fn = padder.collate_fn)
        with torch.no_grad():
            correct_cntr = 0
            total_cntr = 0
            for sample in dev_dataloader:
                batch_data_list, batch_label_list, batch_len_list, padded_sublens = sample

                batch_tag_score = tagger.forward(batch_data_list, batch_len_list, padded_sublens)
              
                flatten_tag, flatten_label = self._flat_vecs(batch_tag_score, batch_label_list)

                #calc accuracy
                c, t = self._calc_batch_acc(tagger, flatten_tag, flatten_label)
                correct_cntr += c 
                total_cntr += t
       
        acc = correct_cntr/total_cntr
        self.acc_data_list.append(acc)
        print("Validation accuracy " + str(acc))
        
        tagger.train()


    def _saveAccData(self):
        try:
            acc_data = torch.load('accuracy_graphs_data')
        except FileNotFoundError:
            print("No accuracy data file found - creating new")
            acc_data = {}

        acc_data.update({str('ModelParams'): self.acc_data_list})
        torch.save(acc_data, 'accuracy_graphs_data')

    def test(self):
        test_dataset = As3Dataset(file_path = self.test_file, 
                                  is_test_data = True, is_train_data = False)

        self.wTran = WTranslator(None, None, None, None, False)
        self.lTran = TagTranslator(None, False)

        self._load_translators_params(self.wTran, self.lTran)
        #test_dataset.toIndexes(wT = self.wTran, lT = self.lTran)
        test_dataset.setTranslators(wT = self.wTran, lT = self.lTran)

        tagger = BiLSTM(embedding_dim = self.edim, hidden_rnn_dim = self.rnn_h_dim,
                translator=self.wTran, tagset_size = self.lTran.getLengths()['tag'] + 1,
                c_embedding_dim = self.c_embedding_dim, dropout = self.dropout)

        self._load_bilstm_params(tagger)
        padder = Padding(self.wTran, self.lTran)
       
        test_dataloader = DataLoader(dataset=test_dataset,
                          batch_size=1, shuffle=False,
                          collate_fn = padder.collate_fn)

        reversed_dict = reverseDict(self.lTran.tag_dict)
        reversed_dict.append('UNKNOWN')
        with torch.no_grad():
            with open(self.test_o_file, 'w') as wf:
                for sample in test_dataloader:
                    batch_data_list, batch_label_list, batch_len_list, padded_sublens = sample
                    batch_tag_score = tagger.forward(batch_data_list,
                                                     batch_len_list, padded_sublens)
                    for i, sample_tag_list in enumerate(batch_tag_score):
                        predicted_tags = tagger.getLabel(sample_tag_list)
                        for j in range(batch_len_list[i]):
                            t = predicted_tags[j]
                            w = reversed_dict[t]
                            wf.write(str(w) + "\n")
                        wf.write("\n")

    def train(self):
        print("Loading data")
        train_dataset = As3Dataset(self.train_file, is_test_data=False, is_train_data=True)
        print("Done loading data")

        self.wTran = WTranslator()
        self.lTran = TagTranslator()

        print("translate to indexes")
        #train_dataset.toIndexes(wT = self.wTran, lT = self.lTran)
        train_dataset.setTranslators(wT = self.wTran, lT = self.lTran)
        print("done")

        print("init tagger")
        tagger = BiLSTM(embedding_dim = self.edim, hidden_rnn_dim = self.rnn_h_dim,
                translator=self.wTran, tagset_size = self.lTran.getLengths()['tag'] + 1,
                c_embedding_dim = self.c_embedding_dim, dropout = self.dropout)
        print("done")

        print("define loss and optimizer")
        loss_function = nn.CrossEntropyLoss() #ignore_index=len(lTran.tag_dict))
        optimizer = torch.optim.Adam(tagger.parameters(), lr=self.learning_rate) #0.01)
        print("done")

        print("init padder")
        padder = Padding(self.wTran, self.lTran)
        print("done")

        train_dataloader = DataLoader(dataset=train_dataset,
                          batch_size=self.batch_size, shuffle=True,
                          collate_fn = padder.collate_fn)
        
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
                if progress1/1000 == progress2:
                    print("reached " + str(progress2*1000))
                    progress2+=1
                progress1 += self.batch_size
                sentences_seen += self.batch_size

                tagger.zero_grad()
                #batch_data_list, batch_label_list, batch_len_list, padded_sublens = sample
                
                batch_tag_score = tagger.forward(sample)
              
                flatten_tag, flatten_label = self._flat_vecs(batch_tag_score, batch_label_list)

                #calc accuracy
                c, t = self._calc_batch_acc(tagger, flatten_tag, flatten_label)
                correct_cntr += c 
                total_cntr += t

                loss = loss_function(flatten_tag, flatten_label)
                loss_acc += loss.item()
                loss.backward()
                optimizer.step()

                if sentences_seen >= 500:
                    sentences_seen = 0
                    if self.run_dev:
                        self.runOnDev(tagger, padder) 
                
            print("epoch: " + str(epoch) + " " + str(loss_acc))
            print("Train accuracy " + str(correct_cntr/total_cntr))
           
        if self.save_to_file:
            self._save_model_params(tagger, self.wTran, self.lTran)

        if self.run_dev:
            self._saveAccData()
        #if (sys.argv[1] == 'save') or (sys.argv[1] == 'loadsave'):
            #self._save_model_params(tagger, self.wTran, self.lTran)
            #torch.save(tagger.state_dict(), 'bilstm_params.pt')


FAVORITE_RUN_PARAMS = { 
                'EMBEDDING_DIM' : 50, 
                'RNN_H_DIM' : 50, 
                'EPOCHS' : 20, 
                'BATCH_SIZE' : 2,
                'CHAR_EMBEDDING_DIM': 15,
                'LEARNING_RATE' : 0.01
                }

if __name__ == "__main__": 
    train_file = "./data/snli_1.0/small_dataset.jsonl"
                 #"sys.argv[1]
    model_file = 'SOMEMODEL' #sys.argv[2]
    epochs = 1 #int(sys.argv[3])
    run_dev = 'n' #sys.argv[4]
    if run_dev == 'y':
        run_dev = True
        dev_file = sys.argv[5]
    else:
        run_dev = False
        dev_file = None
   
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
