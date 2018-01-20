#import jieba
import re
import json
import codecs
import string
import random
import numpy as np
import _pickle as cPickle
import torch
# s = codecs.open('ResearchTrainSample.json', mode='r', encoding='utf-8-sig').read()
# codecs.open('ResearchTrainSample.json', mode='w', encoding='utf-8').write(s)


is_cuda =False #torch.cuda.is_available()


f = codecs.open('data/ResearchTrainSample.json', mode='r', encoding='utf-8')
reports = json.load(f)

with open('data/ChineseChars.pkl', 'rb') as handle:
    chinesechars = cPickle.load(handle)


filters = '!,?;；，！？。：:\r\n'
split = "|"
fp = codecs.open('data/strategy_tag.txt', 'a', encoding='utf-8')
result = []

vocabs = []
for report in reports:
    if 'content' in report:
        text = report['content'].replace('|', "").replace('', '').replace('\u3000', "")
        vocabs.extend(sorted(list(set(text))))
vocabs = sorted(list(set(vocabs)))
vocabs.insert(0, '<unk>')  # 2  unknown
vocabs.insert(0, '<eos>')  # 1 end of sentance
vocabs.insert(0, '<sos>')  # 0 start ofsentance


print('vocabs:{}'.format(len(vocabs)))
_idx2char = {i: w for i, w in enumerate(vocabs)}
_char2idx = {w: i for i, w in enumerate(vocabs)}


def char2idx(charstr):
    if charstr in _char2idx:
        return _char2idx[charstr]
    else:
        return 2


def idx2char(idx):
    if idx in _idx2char:
        return _idx2char[idx]
    else:
        return '<unk>'


# 找出潛在標籤
def prepare_tags():
    for report in reports:
        if 'content' in report:
            text = report['content'].replace('|', "").replace('', '').replace('\u3000', "")
            translate_map = str.maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
            seq = text.split(split)
            seq = list(filter(None, seq))
            for item in seq:
                # if '评级' in item or '目标价' in item or '买入' in item:
                if 'eps' in str.lower(item) or 'EPS' in item or '每股盈余' in item:
                    result.append(report['news_id'] + '\t' + item + '\r\n')
                    if len(result) >= 100:
                        fp.writelines(result)
                        result.clear()
    fp.writelines(result)
    result.clear()


seg_methods = ['onehot', 'character', 'word2vec']
def sentance2seq(sentance, seg_method='onehot',max_length=50):
    results_arr = []
    if seg_method == 'onehot':
        seq_x = [char2idx(tok) for tok in list(sentance)]
        while len(seq_x)<max_length:
            seq_x.append(1)
        #results_arr = np.eye(len(vocabs), dtype=np.float32)[seq_x]
        return seq_x
    elif seg_method == 'character':
        seq_x = [np.reshape(chinesechars[tok],-1) for tok in list(sentance)]
        while len(seq_x)<max_length:
            seq_x.append(np.random.standard_normal(1089))
        return seq_x


# 0: 無特殊標籤
# 1: 投資策略
class sequence_reader(object):
    def __init__(self, file_path='data/ResearchTrainSample.json',max_length=50,is_train=True,is_onehot=False):
        '''
		負責載入完整end-to-end中文建模需要之語料.
		'''
        self.file_path = file_path
        self.is_train=is_train
        self.is_onehot=is_onehot
        self.is_train=is_train
        self.max_length=max_length
        f = codecs.open(self.file_path, mode='r', encoding='utf-8')
        reports = json.load(f)
        f1 = codecs.open('data/strategy_tag.txt', mode='r', encoding='utf-8')
        self.taggeddict = {}

        strategies = f1.readlines()
        # 將標籤依照newid做規整，存放在對應list中
        for strategy in strategies:
            if strategy.split('\t')[0] not in self.taggeddict:
                self.taggeddict[strategy.split('\t')[0]] = []
                self.taggeddict[strategy.split('\t')[0]].append(strategy.split('\t')[1])
            else:
                self.taggeddict[strategy.split('\t')[0]].append(strategy.split('\t')[1])
        #把標注結果放在self.taggeddict
        if self.is_train:
            self.sequence_lines = [[r['news_id'], r['content'].replace('|', "").replace('', '').replace('\u3000', "")]for r in reports if 'content' in r and r['news_id'] in self.taggeddict ]
        else:
            self.sequence_lines=[[r['news_id'],r['content'].replace('|', "").replace('', '').replace('\u3000', "")] for r in reports if 'content' in r ]
        self.idx = 0
        self.feasures = []
        self.labels = []
        self.groundtruth = []

    def size(self):
        return len(self.sequence_lines)
    def has_more(self):
        if self.idx < self.size() - 1:
            return True
        return False
    def reset(self):
        self.idx = 0
        np.random.shuffle(self.sequence_lines)

    def current_minibatch(self):
        return self.feasures

    def next_minibatch(self,minibatch_size:int):
        global is_onehot
        '''
		Return a mini batch of sequence frames and their corresponding ground truth.
		'''
        batch_x = []
        batch_y = []

        while len(batch_x)<minibatch_size:
            text=self.sequence_lines[self.idx][1]
            translate_map = str.maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
            seqs = text.split(split)
            seqs = list(filter(None, seqs))
            for seq in seqs:
                # 此時已經把文章切成一段一段
                #如果新聞id不在標籤列表中
                if len(seq)>self.max_length-1:
                    seq=seq[:self.max_length]
                if self.sequence_lines[self.idx][0] not in  self.taggeddict:
                        y =[]
                        [y.append(0) for i in  range(min(len(seq),self.max_length))]
                        while len(y)<self.max_length:
                            y.append(0)
                        #batch_y.append(np.eye(2, dtype=np.float32)[y])
                        if not self.is_onehot:
                            batch_y.append(y)
                            batch_x.append(sentance2seq(seq))
                        else:
                            batch_y.append(np.eye(2, dtype=np.float32)[y])
                            batch_x.append(np.eye(len(vocabs), dtype=np.float32)[sentance2seq(seq)])
                # 如果新聞id在標籤列表中
                else:
                        strategies_set = self.taggeddict[self.sequence_lines[self.idx][0]]
                        y = []
                        [y.append(0) for i in range(min(len(seq),self.max_length))]
                        while len(y)<self.max_length:
                            y.append(0)
                        for s in strategies_set:
                            test=seq.find(s)
                            if seq.find(s)!=-1:
                                y[seq.find(s):seq.find(s)+len(s)]=1

                        if not self.is_onehot:
                            batch_y.append(y)
                            batch_x.append(sentance2seq(seq))
                        else:
                            batch_y.append(np.eye(2, dtype=np.float32)[y])
                            batch_x.append(np.eye(len(vocabs), dtype=np.float32)[sentance2seq(seq)])

            self.idx=random.randint(0,len(self.sequence_lines))
            if self.idx>len(self.sequence_lines)-1:
                self.idx=0
            return batch_x, batch_y

if __name__ == '__main__':
    #prepare_tags()
    reader=sequence_reader()
    x,y=reader.next_minibatch(16)