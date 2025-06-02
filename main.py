import csv
import json
import os
import random
import time

# import pandas as pd
import torch
from jsonlines import jsonlines
from nltk import WordPunctTokenizer

# from Classifier.bert.data_utils import get_tag2idx
# from Classifier.bert.train import train
from Classifier.bart.bart_test import bart_test
from Classifier.bart.bart_train import bart_train
from Classifier.bart.integrate_test import integrate_test
from Classifier.bart.small_bart_test import small_bart_test
from Classifier.bart.small_bart_train import small_bart_train
from config import get_config
import numpy as np

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed_all(seed+2)
    np.random.seed(seed+3)
    os.environ['PYTHONHASHSEED']=str(seed+4)
    torch.backends.cudnn.deterministic=True

class DataProcessor:
    def __init__(self,args,logger):
        self.args=args
        self.logger = logger

    def to_tokens(self,s: str): #将句子转化为token列表
        word_tokenizer=WordPunctTokenizer()
        return word_tokenizer.tokenize(s)
    def local_in_token(self,tokens: list, trigger: str): #返回trigger在token列表中的位置（如要进行切片操作须在返回值后append末位值+1）
        if len(self.to_tokens(trigger)) == 1:
            return [tokens.index(trigger)]
        else:
            trlist = self.to_tokens(trigger)
            i = 0
            j = 0
            # print(tokens)
            # print(trlist)
            while i < len(tokens):
                while j <= len(trlist):
                    if tokens[i] == trlist[j]:
                        i += 1
                        j += 1
                    else:
                        j = 0
                        i += 1
                    if j == len(trlist):
                        relist = []
                        for o in range(len(trlist)):
                            relist.append(i - len(trlist) + o)
                        return relist
    def View_sentence(self,dic: dict): #传入生成格式的triplet的值，返回sentence,head,tail,label
        s = ""
        head = ""
        tail = ""
        tokenlist = dic['tokens']
        headindex = dic['head']
        tailindex = dic['tail']
        label = dic['label'] if dic['label'] != 'None' else 'N'
        for i in tokenlist:
            s = s + i + ' '
        for i in headindex:
            head = head + tokenlist[i] + ' '
        for i in tailindex:
            tail = tail + tokenlist[i] + ' '
        return s, head, tail, label
    def create_balance_Train_and_Dev(self,opath, trpath, depath): #对生成模型划分训练集和测试集
        all = []
        with jsonlines.open(opath) as reader:
            for obj in reader:
                all.append(obj)
        tr = [i for i in all if i["triplets"][0]["label"] != 'bug']
        labelset = list(set(i["triplets"][0]["label"] for i in tr))
        dlist = {}
        for olabel in labelset:
            dlist[olabel] = [i for i in all if i["triplets"][0]["label"] == olabel]
        length = min([len(dlist[i]) for i in dlist])
        bdlist = {}
        for i in dlist:
            bdlist[i] = random.sample(dlist[i], length)
        with jsonlines.open(trpath, mode='w') as writer:
            for o in bdlist:
                for i in bdlist[o][:int(len(bdlist[o]) * 0.7)]:
                    writer.write(i)
        with jsonlines.open(depath, mode='w') as writer:
            for o in bdlist:
                for i in bdlist[o][int(len(bdlist[o]) * 0.7):]:
                    writer.write(i)
    def split_train_and_dev(self,path,trpath,depath): #为分类数据划分训练集和测试集
        dlist = {}
        if str(path).split(".")[-1]=="json":
            with open(path, 'r', encoding='utf-8') as f:
                oridata = json.load(f)
            labelset=list(set(i["label"] for i in oridata))
            for olabel in labelset:
                dlist[olabel] = [i for i in oridata if i["label"] == olabel]
        trlist = []
        for o in dlist:
            for i in dlist[o][:int(len(dlist[o]) * 0.7)]:
                trlist.append(i)
        jf = open(trpath, mode='w')
        json.dump(trlist, jf, indent=4)
        jf.close()

        delist = []
        for o in dlist:
            for i in dlist[o][int(len(dlist[o]) * 0.7):]:
                delist.append(i)
        jf = open(depath, mode='w')
        json.dump(delist, jf, indent=4)
        jf.close()
        # with jsonlines.open(trpath, mode='w') as writer:
        #     for o in dlist:
        #         for i in dlist[o][:int(len(dlist[o]) * 0.7)]:
        #             writer.write(i)
        # with jsonlines.open(depath, mode='w') as writer:
        #     for o in dlist:
        #         for i in dlist[o][int(len(dlist[o]) * 0.7):]:
        #             writer.write(i)

class MainGenClassifier:
    def __init__(self,args,logger):
        self.args = args
        self.logger = logger
    def run(self):
        datastr = self.args.dataset
        version = self.args.version
        n_epoch=self.args.gen_train_epochs
        path=str(f"Data/{datastr}/")
        trpath=path+"train.json"
        depath=path+"val.json"
        tepath=path+"test.json"
        tagpath=path+"class_map.json"
        save_path=str(f"Data/{datastr}/v{version}_{n_epoch}epoches_result.csv")
        model1path=str(f"SaveModels/{datastr}/vbart/won_model{self.args.model1}.bin")
        model2path=str(f"SaveModels/{datastr}/vsmodel/{self.args.model2}won_model.bin")
        if version=='tim-tr':
            bart_train(self.args, self.logger, trpath=trpath, depath=depath, tagpath=tagpath,save_path=model1path, metric="micro")
        elif version=='tjm-tr':
            small_bart_train(self.args, self.logger, trpath=trpath, depath=tepath, tagpath=tagpath,save_path=model2path)
        elif version=='tim-te':
            bart_test(self.args, self.logger, tepath=tepath, tagpath=tagpath)
        elif version=='tjm-te':
            small_bart_test(self.args, self.logger, trpath=trpath, depath=tepath, tagpath=tagpath)
        elif version=='ftest':
            integrate_test(self.args, self.logger, tepath=tepath, tagpath=tagpath, save_path=save_path,
                           model1path=model1path, model2path=model2path)
if __name__=='__main__':
    # time.sleep(3600*0.5)
    args, logger = get_config()
    seed_everything(args.seed)
    cls=MainGenClassifier(args,logger)
    args.version='tim-tr'
    cls.run()
    args.version = 'tjm-tr'
    cls.run()
    args.version='ftest'
    cls.run()