# coding=gbk
import sys
import os
import pandas as pd
sys.path.append('..')
from src.utils import config
from src.relation import processing,modelMode

#用模型获取True  用规则+模型获取False
useModelOnly = True
files = ['美国-原始数据.xlsx','美国原始数据_新加1.xlsx','美国原始数据_新加2.xlsx'] #11264

def getRelationTriples(files,useModelOnly):
       # test_result_path = os.path.join(config.BASE_DIR, 'save\\relation','test_result.json')
       test_result_path = os.path.join(config.BASE_DIR, 'save\\relation', 'test_result.xlsx')
       triple_path = os.path.join(config.BASE_DIR, 'data\\raw_data', 'triple.xlsx')
       fileList = ["renwu.csv", "didian.csv", "zuzhi.csv"]

       #加载模型
       subject_model,object_model,id2rel,tokenizer = modelMode.loadModel()
       #获取要抽取关系的段落数据
       dataList = processing.getContentData(files)
       #获取规则关系库和关系库中的实体库
       triple_list,entity_list = processing.getTriples(triple_path)
       #获取所有实体库
       entityList = processing.entityList(fileList)

       #逐条处理
       allData = pd.DataFrame()
       if not useModelOnly:
              print("规则加模型")
              for i,content in enumerate(dataList):
                     # 先规则匹配
                     newtext, data1 = processing.reg_relation(triple_list, entity_list,content)
                     allData = allData.append(data1, ignore_index=True)
                     #剩下的句子模型匹配
                     if (len(newtext)!=0):
                            senList = processing.processing(newtext,entityList)
                            data2 = modelMode.metric2excel(subject_model, object_model, senList, id2rel, tokenizer, content)
                            allData = allData.append(data2, ignore_index=True)
                     print("第{}条content关系抽取结束".format(i))
       else:
              print("模型")
              for i, content in enumerate(dataList):
                     senList = processing.processing(content, entityList)
                     data2 = modelMode.metric2excel(subject_model, object_model, senList, id2rel, tokenizer, content)
                     allData = allData.append(data2, ignore_index=True)
                     print("第{}条content关系抽取结束".format(i))

       #保存文件
       F = open(test_result_path, 'a+')
       allData.to_excel(test_result_path,encoding='utf-8',index=False,header=True)
       F.close()

getRelationTriples(files,useModelOnly)