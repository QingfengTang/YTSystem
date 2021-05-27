#coding:utf-8
import torch

from fastNLP import DataSet
from src.intent.getAndSetTree import *
from fastNLP import Vocabulary
import jieba
from nltk.probability import FreqDist
import json
import re
from fastNLP import BucketSampler
from fastNLP import DataSetIter
from src.intent.model  import BiLSTMMaxPoolCls

class IntentDiscovery():
    def __init__(self, json_path, stop_word_path, vocab_path, model_path):
        self.json_data = json_path
        self.stop_word_path = stop_word_path
        self.vocab_path = vocab_path
        self.model_path = model_path

    def build(self):
        raise NotImplementedError('this method implement create model')

    def run(self, title, content):
        # 在content中找高频词，并添加到title中
        with open(self.json_data, "r", encoding="utf-8") as f:
            dict_labels = json.load(f)
        text = content
        words = jieba.lcut(str(text))
        fdict = FreqDist(words)
        delarr = []
        setarr = []
        count = 0
        for key in fdict:
            if len(key) < 2:
                delarr.append(key)
            else:
                with open(self.stop_word_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for sw in lines:
                        temp = sw.strip('\n')
                        if len(temp) < 2:
                            continue
                        else:
                            if key == temp:
                                delarr.append(key)
                                break

        for key in delarr:
            del fdict[key]

        tops = fdict.most_common(50)
        for key in fdict:
            for t_value in dict_labels.values():
                for word in t_value:
                    if key == word:
                        setarr.append(word)

        if len(setarr) > 11:
            setarr = setarr[0:10]
        else:
            t = 10 - len(setarr)
            count = 0
            for key in fdict:
                res = re.match('^[0-9]*$', key)
                if res:
                    continue
                if key not in setarr and count < t:
                    setarr.append(key)
                    count = count + 1
        title = title + "".join(setarr)
        # 数据格式转换
        data = {'title': [title]}
        dataset = DataSet(data)
        dataset.apply_field(list, field_name='title', new_field_name='chars')
        dataset.apply(lambda ins: len(ins['chars']), new_field_name='seq_len')
        dataset.set_input('chars', 'seq_len')

        vocab = Vocabulary()
        #  从该dataset中的chars列建立词表,验证集或者测试集在建立词表是放入no_create_entry_dataset这个参数中。
        vocab.load(self.vocab_path)
        vocab.from_dataset(dataset, field_name='chars')
        vocab.index_dataset(dataset, field_name='chars')

        dic_lable = {'0': ['控制疫情_美国'], '1': ['扰乱国际秩序'], '2': ['政治斗争'], '3': ['军事干涉世界局势'], '4': ['国际疫情政治化'],
                     '5': ['控制暴乱'], '6': ['其他'], '7': ['缓解经济压力_美国'],
                     '8': ['操控国际经济'], '9': ['提高社会保障'], '10': ['借环境问题打压别国']}
        #
        batch_size = 1
        test_sampler = BucketSampler(batch_size=batch_size, seq_len_field_name='seq_len')
        test_batch = DataSetIter(batch_size=batch_size, dataset=dataset, sampler=test_sampler)

        state = torch.load(self.model_path)
        state.eval()
        res = ''
        for b_words, b_target in test_batch:
            output = state(b_words['chars'], b_words['seq_len'])

            output = torch.max(output['pred'], dim=1)[1].numpy()
            output = str(output[0])
            for x in dic_lable:
                if x == output:
                    res = dic_lable[x]
                    break

        result = getChainByName(res[0])
        return result

if __name__ == '__main__':

    intent_disc = IntentDiscovery("../../data/intent_data/json.json",
                    '../../data/stopwords.txt',
                    '../../data/intent_data/vacab.txt',
                                  '../../models/intent_model/best_BiLSTMMaxPoolCls_acc_2021-05-25-22-10-26-355900',
                                  )

    title = '美媒：白宫医疗专家要求彭斯开除新任疫情顾问'
    content = '白宫疫情顾问斯科特·阿特拉斯（图源：路透社） 海外网10月21日电 白宫疫情顾问斯科特·阿特拉斯自8月上任以来，多次发表争议言论，受到国内医疗专家的驳斥。近日，美媒曝出白宫疫情协调专员黛博拉·伯克斯，曾在私下要求副总统彭斯开除阿特拉斯，称自己不相信他。 白宫疫情协调专员黛博拉·伯克斯（图源：路透社） 《华盛顿邮报》文章写道，阿特拉斯来到白宫后，疫情应对小组中的不和谐越发严重。有白宫人士称阿特拉斯了解信息不全面，喜欢掌控大局，甚至是不诚实，还多次向伯克斯、福奇等人强调被医疗专家视为“科学垃圾”的言论。 彭斯领导的疫情应对小组（图源：路透社） 有两位知情人士透露，伯克斯曾为此与领导疫情应对小组的彭斯当面对质。伯克斯对彭斯说道，自己并不相信阿特拉斯，也不相信他提供的是可靠建议，想将他踢出疫情应对小组。然而，在近期的一次对话中，彭斯却让两人自己解决分歧。 美媒指出，阿特拉斯坚持的群体免疫、解除封锁和支持开学言论等早已被医疗专家否定，这也使得福奇和伯克斯等人要求阿特拉斯为自己提供数据支持。上周，阿特拉斯还在社交媒体声称戴口罩无助于防疫，最终被推特删帖。《华盛顿邮报》写道，相比于在美国持续活跃的疫情，白宫疫情应对小组则处于“休眠状态”。（海外网 赵健行）'
    result = intent_disc.run(title, content)
    print(result)