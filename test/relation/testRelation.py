# coding=gbk
import sys
import os
import pandas as pd
sys.path.append('..')
from src.utils import config
from src.relation import processing,modelMode

#��ģ�ͻ�ȡTrue  �ù���+ģ�ͻ�ȡFalse
useModelOnly = True
files = ['����-ԭʼ����.xlsx','����ԭʼ����_�¼�1.xlsx','����ԭʼ����_�¼�2.xlsx'] #11264

def getRelationTriples(files,useModelOnly):
       # test_result_path = os.path.join(config.BASE_DIR, 'save\\relation','test_result.json')
       test_result_path = os.path.join(config.BASE_DIR, 'save\\relation', 'test_result.xlsx')
       triple_path = os.path.join(config.BASE_DIR, 'data\\raw_data', 'triple.xlsx')
       fileList = ["renwu.csv", "didian.csv", "zuzhi.csv"]

       #����ģ��
       subject_model,object_model,id2rel,tokenizer = modelMode.loadModel()
       #��ȡҪ��ȡ��ϵ�Ķ�������
       dataList = processing.getContentData(files)
       #��ȡ�����ϵ��͹�ϵ���е�ʵ���
       triple_list,entity_list = processing.getTriples(triple_path)
       #��ȡ����ʵ���
       entityList = processing.entityList(fileList)

       #��������
       allData = pd.DataFrame()
       if not useModelOnly:
              print("�����ģ��")
              for i,content in enumerate(dataList):
                     # �ȹ���ƥ��
                     newtext, data1 = processing.reg_relation(triple_list, entity_list,content)
                     allData = allData.append(data1, ignore_index=True)
                     #ʣ�µľ���ģ��ƥ��
                     if (len(newtext)!=0):
                            senList = processing.processing(newtext,entityList)
                            data2 = modelMode.metric2excel(subject_model, object_model, senList, id2rel, tokenizer, content)
                            allData = allData.append(data2, ignore_index=True)
                     print("��{}��content��ϵ��ȡ����".format(i))
       else:
              print("ģ��")
              for i, content in enumerate(dataList):
                     senList = processing.processing(content, entityList)
                     data2 = modelMode.metric2excel(subject_model, object_model, senList, id2rel, tokenizer, content)
                     allData = allData.append(data2, ignore_index=True)
                     print("��{}��content��ϵ��ȡ����".format(i))

       #�����ļ�
       F = open(test_result_path, 'a+')
       allData.to_excel(test_result_path,encoding='utf-8',index=False,header=True)
       F.close()

getRelationTriples(files,useModelOnly)