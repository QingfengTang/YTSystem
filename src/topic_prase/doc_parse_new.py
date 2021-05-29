# -*- coding:utf-8 -*-
import pandas as pd
import json
from LAC import LAC
# from .triggers_utils import *
from src.topic_prase.triggers_utils import *
import nltk
import itertools
import pyltp
from src.utils.functions import document2sentences
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from src.utils.config import BASE_DIR
class DocumentAnalyze(object):
    def __init__(self):
        # 抽取实体
        self.lac = LAC(mode='lac')
        self.organizations = []
        self.names = []
        self.locations = []
        self.times = []
        # 抽取触发词 模型地址
        self.LTP_DATA_DIR = os.path.join(BASE_DIR, 'models\\topic_prase\\ltp_data_v3.4.0')# ltp模型目录的路径
        self.trigger_stw_DIR = os.path.join(BASE_DIR, 'data\\trigger_stopword.txt')
        self.data_DIR = os.path.join(BASE_DIR, 'save\\topic\\topic_discovery.xlsx')
        self.triggers = []
        # 结果保存路径
        # self.save_path = '../data/parse_result.csv'
        self.entity2trigger = {}
        self.name2trigger = {}
        self.org2trigger = {}
        ############################
        # 这部分写到excel表格里面
        self.contentid = []
        self.senid = []
        self.triggers2exc = []
        self.name2exc = []
        self.org2exc = []
        ####################
        # 这部分记录编号每个句子的触发词，人物，组织及其起始位置和末位置
        self.triggers_record = {}
        self.names_record = {}
        self.org_record = {}
        ####################
        self.topic_info = pd.DataFrame()
    def build(self,topic_id):
        self.extra_topic(topic_id=topic_id)
        self.triggers_filter()
        self.creat_connection()
        self.save_data()
    def get_triggerstw(self):
        with open(self.trigger_stw_DIR,'r',encoding='gbk') as f:
            lines = f.readlines()
            stop_words = []
            for line in lines:
                word = line.strip()
                stop_words.append(word)
        return stop_words
    def extra_topic(self,topic_id):
        self.topic_id = topic_id
        df = pd.read_excel(self.data_DIR)
        # 筛选某一话题的content
        topic_data = df.loc[df['topic_id'] == topic_id]
        for index, row in topic_data.iterrows():
            # content_index_list.append(index)
            sentences = document2sentences(row['content'])
            sentence_id = 1
            for sentence in sentences:
                self.extra_sentence(sentence,is_trigger=True,content_id= index,sentence_id = sentence_id)
                sentence_id += 1
    def extra_sentence(self, sentence, is_trigger,content_id, sentence_id):
        """
                抽取文档中的 人，组织和地点
                :param sentences: 句子集合
                :param is_trigger: 是否抽触发词
                :return: None
        """
        names = []
        organizations = []
        triggers_temp = []
        triggers = []
        locations = []
        lac_result = self.lac.run(sentence)
        # print(lac_result)
        for i in range(len(lac_result[0])):
            if lac_result[1][i] == 'nt' or lac_result[1][i] == 'ORG':
                organizations.append(lac_result[0][i])
            elif lac_result[1][i] == 'ns' or lac_result[1][i] == 'LOC':
                locations.append(lac_result[0][i])
            elif lac_result[1][i] == 'nr' or lac_result[1][i] == 'PER':
                names.append(lac_result[0][i])
            else:
                continue
        if is_trigger:
            triggers_temp = extract_nvn(sentence)
        organizations = list(set(organizations))
        names = list(set(names))
        if is_trigger:
            triggers = list(set(triggers_temp))
            # 获得一句话的触发词triggers['窃取数据', '表达担心']
            # print(triggers)
        # 找寻与这个触发词的位置信息
        # 句子里面没有触发词
        if len(triggers) == 0:
            return
        # 记录这个句子的各个词的位置信息
        """
        {
            triggers:{ trigger1:{   start:
                                    end:  }}
        }
        """
        # 句子层信息
        this_sentence_record = {}
        # 触发词层信息
        triggers_record = {}
        for trigger in triggers:
            this_trigger_value = {}
            start = sentence.find(trigger)
            end = start + len(trigger)
            this_trigger_value['start'] = start
            this_trigger_value['end'] = end
            triggers_record[trigger] = this_trigger_value
        if content_id in self.triggers_record:
            self.triggers_record[content_id][sentence_id] = triggers_record
        else:
            self.triggers_record[content_id] = {}
            self.triggers_record[content_id][sentence_id] = triggers_record

        # 名字层信息
        names_record = {}
        for name in names:
            this_name_value = {}
            start = sentence.find(name)
            end = start + len(name)
            this_name_value['start'] = start
            this_name_value['end'] = end
            names_record[name] = this_name_value
        if names_record != {}:
            if content_id in self.names_record:
                self.names_record[content_id][sentence_id] = names_record
            else:
                self.names_record[content_id] = {}
                self.names_record[content_id][sentence_id] = names_record
        # 组织层信息
        orgs_record = {}
        for org in organizations:
            this_org_value = {}
            start = sentence.find(org)
            end = start + len(org)
            this_org_value['start'] = start
            this_org_value['end'] = end
            orgs_record[org] = this_org_value
        if orgs_record != {}:
            if content_id in self.org_record:
                self.org_record[content_id][sentence_id] = orgs_record
            else:
                self.org_record[content_id] = {}
                self.org_record[content_id][sentence_id] = orgs_record

    def triggers_filter(self):
        triggers_record = self.triggers_record
        # 统计触发词词频
        trigger_temp = []
        for content_id in triggers_record:
            for sentence_id in triggers_record[content_id]:
                for trigger in triggers_record[content_id][sentence_id]:
                    trigger_temp.append(trigger)
        freq_dist = nltk.FreqDist(trigger_temp)
        triggers_freq = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)
        list_triger = []
        list_freq = []
        list_trigers = []
        trigger_stw = self.get_triggerstw()
        for k, v in triggers_freq:
            k = k.strip()
            list_triger.append(k)
            list_freq.append(v)
            if v>2 and k not in trigger_stw and len(k) >=4:
                list_trigers.append(k)
        dict = {
            '触发词': list_triger,
            '频率': list_freq
        }
        df = pd.DataFrame(dict)
        # df.to_excel(r'C:\Users\xuchanghua\PycharmProjects\YTCodebase\src\topic_parse\output\trigger_freq.xlsx')
        # 得到词频大于1的触发词列表
        # 对原词频较小的触发词进行删除
        # 构建新的self.triggers_record
        new_dict = {}
        for content_id in triggers_record:
            for sentence_id in triggers_record[content_id]:
                for trigger in triggers_record[content_id][sentence_id]:
                    if trigger in list_trigers:
                        if content_id in new_dict:
                            if sentence_id in new_dict[content_id]:
                                trigger_value = triggers_record[content_id][sentence_id][trigger]
                                new_dict[content_id][sentence_id][trigger] =  trigger_value
                            else:
                                new_dict[content_id][sentence_id] = {}
                                trigger_value = triggers_record[content_id][sentence_id][trigger]
                                new_dict[content_id][sentence_id][trigger] = trigger_value
                        else:
                            new_dict[content_id] = {}
                            new_dict[content_id][sentence_id] = {}
                            trigger_value = triggers_record[content_id][sentence_id][trigger]
                            new_dict[content_id][sentence_id][trigger] = trigger_value

        self.triggers_record = new_dict

    def creat_connection(self):
        for content_id in self.triggers_record:
            for sentence_id in self.triggers_record[content_id]:
                # 这个句子包含的信息：
                triggers_info = self.triggers_record[content_id][sentence_id]
                names_info = {}
                if content_id in self.names_record:
                    if sentence_id in self.names_record[content_id]:
                        names_info = self.names_record[content_id][sentence_id]
                orgs_info = {}
                if content_id in self.org_record:
                    if sentence_id in self.org_record[content_id]:
                        orgs_info = self.org_record[content_id][sentence_id]
                # 这个句子触发词对应的实体
                trigger2entity = {}
                for trigger in triggers_info:
                    trigger2entity[trigger] = {}
                # 组织对应的触发词
                if orgs_info != {}:
                    for org in orgs_info:
                        org_trigger_distance = {}
                        org_start = orgs_info[org]['start']
                        org_end = orgs_info[org]['end']
                        for trigger in triggers_info:
                            trigger_start = triggers_info[trigger]['start']
                            trigger_end = triggers_info[trigger]['end']
                            # 组织在前
                            if org_start < trigger_start:
                                distance = abs(trigger_start - org_end)
                                org_trigger_distance[trigger] = distance
                            else:
                                distance = abs(org_start - trigger_end)
                                org_trigger_distance[trigger] = distance
                        finaltrigger = ''
                        min_distance = 1000
                        for trigger in org_trigger_distance:
                            if org_trigger_distance[trigger] < min_distance:
                                finaltrigger = trigger
                                min_distance = org_trigger_distance[trigger]
                        # 最终实体与触发词为 org 和 finaltrigger
                        if 'org' in trigger2entity[finaltrigger]:
                            trigger2entity[finaltrigger]['org'] += ',' + org
                        else:
                            trigger2entity[finaltrigger]['org'] = org
                        # if org in self.org2trigger:
                        #     if finaltrigger not in self.org2trigger[org]:
                        #         self.org2trigger[org].append(finaltrigger)
                        # else:
                        #     self.org2trigger[org] = []
                        #     self.org2trigger[org].append(finaltrigger)
                # 人名对应的触发词
                if names_info != {}:
                    for name in names_info:
                        name_trigger_distance = {}
                        name_start = names_info[name]['start']
                        name_end = names_info[name]['end']
                        for trigger in triggers_info:
                            trigger_start = triggers_info[trigger]['start']
                            trigger_end = triggers_info[trigger]['end']
                            # 人名在前
                            if name_start < trigger_start:
                                distance = abs(trigger_start - name_end)
                                name_trigger_distance[trigger] = distance
                            else:
                                distance = abs(name_start - trigger_end)
                                name_trigger_distance[trigger] = distance
                        finaltrigger = ''
                        min_distance = 1000
                        for trigger in name_trigger_distance:
                            if name_trigger_distance[trigger] < min_distance:
                                finaltrigger = trigger
                                min_distance = name_trigger_distance[trigger]
                        # 最终实体与触发词为 name 和 finaltrigger
                        if 'name' in trigger2entity[finaltrigger]:
                            trigger2entity[finaltrigger]['name'] += ',' + name
                        else:
                            trigger2entity[finaltrigger]['name'] = name
                        # if name in self.name2trigger:
                        #     if finaltrigger not in self.name2trigger[name]:
                        #         self.name2trigger[name].append(finaltrigger)
                        # else:
                        #     self.name2trigger[name] = []
                        #     self.name2trigger[name].append(finaltrigger)
                for trigger in trigger2entity:
                    self.contentid.append(content_id)
                    self.senid.append(sentence_id)
                    self.triggers2exc.append(trigger)
                    if 'name' in trigger2entity[trigger]:
                        self.name2exc.append(trigger2entity[trigger]['name'])
                    else:
                        self.name2exc.append(None)
                    if 'org' in trigger2entity[trigger]:
                        self.org2exc.append(trigger2entity[trigger]['org'])
                    else:
                        self.org2exc.append(None)
    def save_data(self):
        save_DIR = os.path.join(BASE_DIR,'save\\topic_prase\\topic_info.xlsx')
        inf = {
                'content_id':self.contentid,
                'sent_id':self.senid,
                'trigger':self.triggers2exc,
                'name':self.name2exc,
                'org': self.org2exc
                }
        self.topic_info = pd.DataFrame(inf)
        self.topic_info.to_excel(save_DIR,index=False)
    def get_relation(self):
        entity2entity = {}
        entity_relation_DIR = os.path.join(BASE_DIR,'data\\relation\\entity_relation.xlsx')
        df = pd.read_excel(entity_relation_DIR, sheet_name='Sheet1')
        for index, row in df.iterrows():
            entity1 = row['entity1']
            entity2 = row['entity2']
            # if entity1 == '英国' or entity2 == '英国':
            #     pass
            if entity1 not in entity2entity:
                entity2entity[entity1] = []
                entity2entity[entity1].append(entity2)
            else:
                if entity2 not in entity2entity[entity1]:
                    entity2entity[entity1].append(entity2)
            if entity2 not in entity2entity:
                entity2entity[entity2] = []
                entity2entity[entity2].append(entity1)
            else:
                if entity1 not in entity2entity[entity2]:
                    entity2entity[entity2].append(entity1)
        return entity2entity

    def run(self,connection_level ,id = int()):
        df = self.topic_info
        if connection_level == 'topic' and id != self.topic_id:
            print('topic_id must be the same number as extra_topic info!')
            return
        data_info = df
        if connection_level == 'topic':
            # topic_id = id
            # data_info = df.loc[df['topic'] == topic_id]
            pass
        elif connection_level == 'content':
            content_id = id
            data_info = df.loc[df['content_id'] == content_id]
        else:
            print('erro connection_level')
            return
        # 创建事件id对应的事件
        id2trigger = {}
        # 创建人物对应的事件
        # 创建组织对应的事件
        # 人物组织之间的关系
        entity2trigger = {}
        entity2entity = {}
        for index, row in data_info.iterrows():
            content_id = row['content_id']
            sentence_id = row['sent_id']
            trigger = row['trigger']
            names = str(row['name'])
            orgs = str(row['org'])
            if content_id in id2trigger:
                if sentence_id in id2trigger[content_id]:
                    id2trigger[content_id][sentence_id].append(trigger)
                else:
                    id2trigger[content_id][sentence_id] = []
                    id2trigger[content_id][sentence_id].append(trigger)
            else:
                id2trigger[content_id] = {}
                id2trigger[content_id][sentence_id] = []
                id2trigger[content_id][sentence_id].append(trigger)
            if names != 'None':
                name_list = names.replace('\n','').split(',')
                for name in name_list:
                    if name in entity2trigger:
                        if trigger not in entity2trigger[name]:
                            entity2trigger[name].append(trigger)
                        else:
                            pass
                    else:
                        entity2trigger[name] = []
                        entity2trigger[name].append(trigger)

            if orgs != 'None':
                org_list = orgs.replace('\n','').split(',')
                for org in org_list:
                    if org in entity2trigger:
                        if trigger not in entity2trigger[org]:
                            entity2trigger[org].append(trigger)
                        else:
                            pass
                    else:
                        entity2trigger[org] = []
                        entity2trigger[org].append(trigger)
        trigger2trigger = {}
        # {
        #     trigger1:{trigger2:0.8
        #                trigger3:0.2
        #               }
        # }
        # 对同一句子建立强度关系
        for content_id in id2trigger:
           for sentence_id in id2trigger[content_id]:
               len_triggers = len(id2trigger[content_id][sentence_id])
               triggers = id2trigger[content_id][sentence_id]
               # 两两枚举
               slect_trigger = list(itertools.permutations(triggers, 2))
               for trigger_pair in slect_trigger:
                   trigger_1 = trigger_pair[0]
                   trigger_2 = trigger_pair[1]
                   if trigger_1 in trigger2trigger:
                       trigger2trigger[trigger_1][trigger_2] = 0.8
                   else:
                       trigger2trigger[trigger_1] = {}
                       trigger2trigger[trigger_1][trigger_2] = 0.8
                   if trigger_2 in trigger2trigger:
                       trigger2trigger[trigger_2][trigger_1] = 0.8
                   else:
                       trigger2trigger[trigger_2] = {}
                       trigger2trigger[trigger_2][trigger_1] = 0.8
        entity2entity = self.get_relation()
        if '' in entity2trigger:
            entity2trigger.pop('')
        # 同一人参与的不同事件
        for entity in entity2trigger:
            # 一个实体参与多个事件
            if len(entity2trigger[entity]) > 1:
                # 两两枚举
                slect_trigger = list(itertools.permutations(entity2trigger[entity], 2))
                for trigger_pair in slect_trigger:
                    trigger_1 = trigger_pair[0]
                    trigger_2 = trigger_pair[1]
                    if trigger_1 in trigger2trigger:
                        if trigger_2 not in trigger2trigger[trigger_1]:
                            trigger2trigger[trigger_1][trigger_2] = 0.5
                        else:
                            trigger2trigger[trigger_1][trigger_2] = max(trigger2trigger[trigger_1][trigger_2],0.5)
                    else:
                        trigger2trigger[trigger_1] = {}
                        trigger2trigger[trigger_1][trigger_2] = 0.5

                    if trigger_2 in trigger2trigger:
                        if trigger_1 not in trigger2trigger[trigger_2]:
                            trigger2trigger[trigger_2][trigger_1] = 0.5
                        else:
                            trigger2trigger[trigger_2][trigger_1] = max(trigger2trigger[trigger_2][trigger_1],0.5)
                    else:
                        trigger2trigger[trigger_2] = {}
                        trigger2trigger[trigger_2][trigger_1] = 0.5
                # 密切相关的实体参与不同的事件
            if entity in entity2entity:
                # 和当前实体密切相关的实体
                list_entity = entity2entity[entity]
                for entity_relate in list_entity:
                    if entity_relate in entity2trigger:
                        # 和这个实体相关的事件大于1
                        if len(entity2trigger[entity])>1:
                            slect_trigger = list(itertools.permutations(entity2trigger[entity], 2))
                            for trigger_pair in slect_trigger:
                                trigger_1 = trigger_pair[0]
                                trigger_2 = trigger_pair[1]
                                if trigger_1 in trigger2trigger:
                                    if trigger_2 not in trigger2trigger[trigger_1]:
                                        trigger2trigger[trigger_1][trigger_2] = 0.2
                                    else:
                                        # trigger2trigger[trigger_1][trigger_2] = max(
                                        #     trigger2trigger[trigger_1][trigger_2], 0.2)
                                        trigger2trigger[trigger_1][trigger_2] = 0.2
                                else:
                                    trigger2trigger[trigger_1] = {}
                                    trigger2trigger[trigger_1][trigger_2] = 0.2

                                if trigger_2 in trigger2trigger:
                                    if trigger_1 not in trigger2trigger[trigger_2]:
                                        trigger2trigger[trigger_2][trigger_1] = 0.2
                                    else:
                                        trigger2trigger[trigger_2][trigger_1] = max(
                                            trigger2trigger[trigger_2][trigger_1], 0.2)
                                else:
                                    trigger2trigger[trigger_2] = {}
                                    trigger2trigger[trigger_2][trigger_1] = 0.2
        trigger_1_list = []
        trigger_2_list = []
        value_list = []
        for trigger_1 in trigger2trigger:
            for trigger_2 in trigger2trigger[trigger_1]:
                trigger_1_list.append(trigger_1)
                trigger_2_list.append(trigger_2)
                value_list.append(trigger2trigger[trigger_1][trigger_2])
        df = pd.DataFrame(
            {
                '触发词1':trigger_1_list,
                '触发词2':trigger_2_list,
                '关系强度':value_list
            }
                        )
        result_DIR = os.path.join(BASE_DIR,'save\\topic_prase\\result.xlsx')
        df.to_excel(result_DIR,index=False)

if __name__ == "__main__":
    DA = DocumentAnalyze()
    topic_id = int(input('话题索引'))
    DA.build(topic_id=topic_id)
    id = int(input('id'))
    connection_level = input('content or topic?')
    DA.run(connection_level ='topic',id=id)
