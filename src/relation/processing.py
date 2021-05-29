import sys
import os,json
import pandas as pd
sys.path.append('..')
from src.utils import functions,config

#获取实体列表
def entityList(files):  #6826个实体 去重后6641
    entityList = []
    for file in files:
        data = pd.read_csv(os.path.join(config.BASE_DIR,'data\\raw_data',file), encoding='utf-8')
        entityList.extend(data['实体名称'].tolist())
    #去重
    newList = list(set(entityList))
    return newList

#匹配实体字符串
def find_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

def processing(text,entityList):
    #先分句
    senList = functions.document2sentences(text)
    #筛选含2个实体的句子
    newSenList = []
    for sen in senList:
        num =0
        enlist = [] #识别出的实体
        for en in entityList:
            if find_idx(sen,en) > -1:
                num +=1
                enlist.append(en)
        if num>1:
            newSenList.append(sen)
    #获得符合要求的句子列表
    return newSenList

def getContentData(files):
    contentList = []
    for file in files:
        file = os.path.join(config.BASE_DIR, 'data\\raw_data', file)
        data = pd.read_excel(file, engine='openpyxl')
        contentList.extend(data['content'].tolist())
    contentList = list(set(contentList))
    print(str(len(contentList))+'条content数据获取完~')
    return contentList

def getTriples(triple_path):
    e = pd.read_excel(triple_path, engine='openpyxl')
    e = e.where(e.notnull(), None)

    # 建立实体库
    # 建立三元组库
    triple_list = []
    entity_list = []

    for i in range(e.values.shape[0]):
        values_i = e.loc[i].values
        triple_tuple = (values_i[0], values_i[1], values_i[2])
        if values_i[0] not in entity_list:
            entity_list.append(values_i[0])
        if values_i[1] not in entity_list:
            entity_list.append(values_i[1])
        if (((values_i[0], values_i[1], values_i[2]) not in triple_list) or (
                (values_i[1], values_i[0], values_i[2]) not in triple_list)):
            triple_list.append(triple_tuple)
    return triple_list,entity_list


# 规则匹配关系
def reg_relation(triple_list, entity_list, text):
    senList = functions.document2sentences(text)
    new_text = ""
    data = pd.DataFrame()
    for sen in senList:
        Pred_triples = set()
        str_entity = []
        for entity in entity_list:
            if entity in sen:
                str_entity.append(entity)
        for i in range(len(str_entity)):
            entity1 = str_entity[i]
            for j in range(i + 1, len(str_entity)):
                entity2 = str_entity[j]
                for k in range(len(triple_list)):
                    triple = triple_list[k]
                    if (triple[0] == entity1 and triple[1] == entity2) or (triple[0] == entity2 and triple[1] == entity1):
                        Pred_triples.add(triple)

        if (len(Pred_triples) != 0):
            for triple in Pred_triples:
                newline = {}
                newline['entity1'] = triple[0]
                newline['entity2'] = triple[1]
                newline['relation'] = triple[2]
                newline['line'] = sen
                newline['content'] = text
                newdata = pd.DataFrame.from_dict(newline, orient='index').T
                data = data.append(newdata, ignore_index=True)
                # newdata.to_excel(output_path, encoding='utf-8', index=False, header=True)
            new_text += ""
        else:
            new_text += sen

    return new_text,data




