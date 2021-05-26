

import pandas as pd
from jieba.analyse import extract_tags,textrank
import re

from src.topic.MyCommunityBuilder import MyCommunityBuilder

'''
    封装话题发现
'''
class TopicDiscover():
    def __init__(self,excel_urls,content_col_name,id_col=None,text_num=100,min_docs=50,stop_word_url=None):
        self.cb = MyCommunityBuilder(min_docs)
        self.splited_df = None
        self.content_col_name = content_col_name
        self.id_col = id_col
        self.splited = False
        self.text_num = text_num
        self.stop_word_url = stop_word_url
        self.stopwords = []
        # self.sentence_df = None
        # self.article_keywords = None
        self.stopwordslist()
        self.getDataFromExcel(excel_urls=excel_urls,content_col_name=content_col_name,id_col =id_col)

    def getDataFromExcel(self,excel_urls,content_col_name,id_col =None):
        ''' 从excel中获得数据
        :param excel_urls: excel路径名队列
        :param content_col_name:excel中表示文本内容的列名，
        :param id_col: 整型，表明excel中是否已经存在id，默认为None
        '''
        content_df = None
        self.content_col_name = content_col_name
        for excel_url in excel_urls:# 从excel_urls中加载内容到类中
            file_df = pd.read_excel(excel_url, index_col=id_col)
            content_df = pd.concat([content_df, file_df], axis=0, ignore_index=True)
        # 对每个文本断句并赋予句子id，提取关键词
        self.article_keywords = []
        content_df['article_keywords'] = None
        num = 0
        for index in content_df.index:
            contents_df = content_df[content_col_name]
            try:
                if num > self.text_num: #当提取文本数量超过text_num时跳出循环
                    break
                num+=1
                content = contents_df.loc[index]
                words = []
                pagekeywords = dict()
                for keyword, weight in textrank(content, withWeight=True, topK=20):
                    ########################停用词
                    if keyword in self.stopwords:
                        continue
                    words.append(keyword)
                    pagekeywords[keyword] = str(round(weight,4))
                    ############################
                content_df['article_keywords'].loc[index] = str(pagekeywords).replace('{','').replace(
                    '}', '').replace("'", '')
                # print(index,words,pagekeywords)
                self.article_keywords.append((index, words, pagekeywords))
                # article_num_all += 1  # id数值追加，使得所有文档的不同文章的id不同
            except:
                print(index, '---')

        article_keywords_dict = dict()
        for article_keyword in self.article_keywords:
            article_keywords_dict[article_keyword[0]] = (article_keyword[1], article_keyword[2])

        content_df['topic_id'] = None
        content_df['topic_keywords'] = None


        self.content_df = content_df
        # return content_df

    def get_split_tag(self,split_by):
        if split_by == 'sentence':
            return '。|！|\!|\.|？|\?'
        elif split_by == 'paragraph':
            return '\n'
        else:
            raise Exception(print('目前的切分文本的依据有句子sentence和段落paragraph,如果要通过其他方式切分文本，请自行处理数据'))
        pass


    def split_data(self,split_tag):
        splited_df = pd.DataFrame(
            columns=('article_id', 'splited_id', 'article_splited', 'splited_content', 'splited_keyword'))
        article_keywords = []
        splited_num = 0
        contents_df = self.content_df[self.content_col_name]  # 获得文本所在的列
        num = 0
        stopwords = self.stopwordslist()
        for text_id in self.content_df.index:  # 对每个文本进行遍历
            if num >= self.text_num:
                break
            num += 1
            try:
                content = contents_df.loc[text_id].strip()  # 获取文本
                spliteds = re.split(split_tag, content)  # 开始分句/段
                splited_id = 0
                for splited in spliteds:
                        if len(splited) > 3:
                            splited_num += 1
                            words = []
                            pagekeywords = dict()

                            # 通过关键词提取获得每一句的关键词及其权重，方便话题发现使用
                            for keyword, weight in textrank(splited, withWeight=True, topK=10):
                                ########################停用词
                                if keyword in self.stopwords:
                                    continue
                                words.append(keyword)
                                pagekeywords[keyword] = str(round(weight,4))
                            # article_keywords是话题发现使用的数组
                            article_keywords.append(((text_id, splited_id), words, pagekeywords))
                            pagekeywords = str(pagekeywords).replace('{', '').replace('}', '').replace("'", '')
                            splited_df.loc[splited_num] = [text_id, splited_id, (text_id, splited_id), splited, pagekeywords]
                            splited_id += 1
            except:
                print(text_id,splited_id, '---')
        for article_keyword in article_keywords:
            print(article_keyword)
        splited_df['topic_id'] = None
        splited_df['topic_keywords'] = None
        self.splited_df = splited_df
        self.article_keywords = article_keywords
        self.splited = True

    def build(self, max_division_depth=5, min_doc_num_per_cate=10, min_nodes_per_cate=10,
              top_keyword_num=30,setID=True):
        '''
        :param max_division_depth: 话题发现的最大迭代深度
        :param min_doc_num_per_cate: 每个话题的最小关键词数量
        :param min_nodes_per_cate: 每个话题的最少文本（句子）数量
        :param top_keyword_num: 选取前top_keyword_num个关键词
        :param setID:是否为话题设置话题ID，建议设置
        '''
        comu_group = self.cb.build(self.article_keywords, max_depth=max_division_depth, min_doc_num=min_doc_num_per_cate,
                              min_nodes=min_nodes_per_cate)
        topic_list = []
        if comu_group:
            if len(comu_group) > 1:
                topic_id = 1
                for comu in comu_group:
                    topic = dict()
                    # comu是每一个话题的信息，在这里输出
                    topicName = comu.generate_description(max=30)
                    tName_word = topicName.split(' ')
                    docCount = comu.get_doc_num()
                    # print(tName_word)
                    text = []
                    if (docCount > 0):
                        docs = comu.iterdoc()
                        wordWeightList = comu.top_keywords(n=top_keyword_num)
                        pagekeywords = dict()
                        for wordID, weight in wordWeightList:
                            pagekeywords[wordID] = str(round(weight,4))
                        for doc in docs:
                            docid = doc.get_docid()
                            text.append(docid)
                    if setID==True:
                        topic['topic_id'] = topic_id
                        print(topic['topic_id'])
                    topic_id+=1
                    topic['topic_name'] = tName_word  # 话题的关键词集合
                    topic['keywords'] = pagekeywords  # 话题中每一个关键词及其权重
                    topic['text'] = text  # 话题所对应文章的列表
                    topic_list.append(topic)
        else:
            print("comu_group is None!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.topic_list = topic_list

        if self.splited == False:
            content_df = self.content_df
            for topic in self.topic_list:
                for article_id in topic['text']:
                    index = article_id
                    try:
                        content_df['topic_id'].loc[index] = topic['topic_id']
                        content_df['topic_keywords'].loc[index] = str(topic['keywords']).replace('{', '').replace('}',
                                                                                                                  '').replace(
                            "'", '')
                    except:
                        print(topic['topic_id'], '------------------')
            self.content_df = content_df

        else:
            content_df = self.splited_df
            for topic in self.topic_list:
                for article_id, sentence_id in topic['text']:
                    index = content_df[(content_df.article_splited == (article_id, sentence_id))].index.tolist()[0]
                    try:
                        content_df['topic_id'].loc[index] = topic['topic_id']
                        content_df['topic_keywords'].loc[index] = str(topic['keywords']).replace('{', '').replace('}',
                                                                                                                  '').replace(
                            "'", '')
                    except:
                        print(topic['topic_id'], '------------------')
            self.splited_df = content_df

    def getTopic(self):
        '''
        :return: 返回topic_list数组
        '''
        return self.topic_list

    def toExcel(self,excel_url):
        '''
        :param excel_url: excel路径
        '''
        if self.splited == False:
            self.content_df.to_excel(excel_url, index=False)
        else:
            self.splited_df.to_excel(excel_url, index=False)

    def delete_NoTopic(self):
        '''
        将生成的sentence_df中为分配到话题的句子删除
        :return:
        '''
        if self.splited == False:
            content_df = self.content_df
        else:
            content_df = self.splited_df
        # sentence_df = self.sentence_df
        # print(content_df.columns.values)
        df = pd.DataFrame(
            columns=(content_df.columns.values))
        for index,row in content_df.iterrows():
            if row['topic_id'] == None or row['topic_id'] == '':
                print('delete',index)
                pass
            else:
                df.loc[index] = row
        if self.splited == False:
            self.content_df = df
        else:
            self.splited_df = df

    def stopwordslist(self):
        if self.stop_word_url != None:
            self.stopwords = [line.strip() for line in open(self.stop_word_url, encoding='UTF-8').readlines()]


# if __name__ == '__main__':
#     td = TopicDiscover(['../data/美国-原始数据.xlsx'],id_col=0,content_col_name= 'content', text_num=1000)
#     split_by = td.get_split_tag('paragraph')
#     td.split_data(split_by)
#     td.build(max_division_depth=10, min_doc_num_per_cate=10, min_nodes_per_cate=10,
#               top_keyword_num=20,setID=True)
#     td.delete_NoTopic()
#     td.toExcel('test7.xlsx')
#     print(td.text_num)
#     pass





