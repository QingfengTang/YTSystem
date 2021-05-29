import os

from jieba.analyse import textrank
import pandas as pd
import numpy as np



keywords_file_default = './data/topic_rec/keywords.txt'
topic_file_default = './data/topic_rec/topic_list.txt'
topic_vec_npy_default = './data/topic_rec/topic_vec.npy'
similar_topic_matrix_npy_default = './data/topic_rec/similar_topic_matrix.npy'

class similar_topic_recommending:
    def __init__(self, news_topic_file,
                 keywords_file=keywords_file_default,
                 topic_file=topic_file_default,
                 topic_vec_npy=topic_vec_npy_default,
                 similar_topic_matrix_npy=similar_topic_matrix_npy_default):
        self.news_topic_file = news_topic_file
        self.keywords_file = keywords_file
        self.topic_file = topic_file
        self.topic_vec_npy = topic_vec_npy
        self.similar_topic_matrix_npy = similar_topic_matrix_npy

    def sim_topic_rec_build(self,index_col=None):
        self.set_keywords2(index_col)
        self.set_topic_keywords(index_col)
        self.get_topic_vec2()
        self.topic_sim_numpy = np.load(self.similar_topic_matrix_npy)

    def load_topic_vec_npy(self,topic_vec_npy):
        self.topic_sim_numpy = np.load(topic_vec_npy)
        print('读取npy文件：',topic_vec_npy)

    def sim_topic_rec_run(self,target_topic_id,threshold=0.001):
        if not hasattr(self,"topic_sim_numpy"):
            self.topic_sim_numpy = np.load(self.similar_topic_matrix_npy)
        self.topic_num = self.topic_sim_numpy.shape[0]
        if target_topic_id <=0 or target_topic_id>self.topic_num:
            raise Exception("输入的话题id不合法")
            return False
        result = self.get_similar_topics2(target_topic_id,threshold)
        return result

    def get_topic_num(self):
        if not hasattr(self,"topic_sim_numpy"):
            self.topic_sim_numpy = np.load(self.similar_topic_matrix_npy)
        self.topic_num = self.topic_sim_numpy.shape[0]
        return self.topic_num

    def set_keywords(self,index_col=None):
        keywords_all = []
        file_df = pd.read_excel(self.news_topic_file, index_col=index_col)
        article_keywords = file_df['article_keywords']
        for index in article_keywords.index:
            try:
                keywords_string = article_keywords[index]
                keywords = keywords_string.split(',')
                for keyword_weight in keywords:
                    keyword_weight = keyword_weight.strip()
                    try:
                        word, weight = keyword_weight.split(':')
                        weight = float(weight.strip())
                        if word not in keywords_all:
                            keywords_all.append(word)
                    except:
                        print('---___', keyword_weight)
            except:
                print('---', index)
        keywords_file_open = open(self.keywords_file, 'w+', encoding='utf-8')
        num = 0
        for index, word in enumerate(keywords_all):
            num += 1
            keywords_file_open.write(str(num) + ' ' + word + '\n');
        keywords_file_open.close()

    def set_keywords2(self,index_col=None):
        keywords_all = []
        file_df = pd.read_excel(self.news_topic_file, index_col=index_col)
        # article_keywords = file_df['article_keywords']
        topic_keywords = file_df['topic_keywords']
        for index in topic_keywords.index:
            try:
                keywords_string = topic_keywords[index]
                keywords = keywords_string.split(',')
                for keyword_weight in keywords:
                    keyword_weight = keyword_weight.strip()
                    try:
                        word, weight = keyword_weight.split(':')
                        weight = float(weight.strip())
                        if word not in keywords_all:
                            keywords_all.append(word)
                    except:
                        print('---___', keyword_weight)
            except:
                print('---===', index)
        keywords_file_open = open(self.keywords_file, 'w+', encoding='utf-8')
        num = 0
        for index, word in enumerate(keywords_all):
            num += 1
            keywords_file_open.write(str(num) + ' ' + word + '\n');
        keywords_file_open.close()


    def set_topic_keywords(self,index_col=None):
        file_df = pd.read_excel(self.news_topic_file, index_col=index_col)
        topic_list = {}
        for index in file_df.index:
            try:
                content_df = file_df.loc[index]
                topic_id = int(content_df['topic_id'])
                topic_keywords = content_df['topic_keywords']
                if topic_id not in topic_list:
                    topic_list[topic_id] = topic_keywords
            except:
                print('---...', index)
        topic_file_open = open(self.topic_file, 'w+', encoding='utf-8')
        for id in topic_list:
            topic_file_open.write(str(id) + '---' + str(topic_list[id]) + '\n')
        topic_file_open.close()

    def get_topic_vec(self):
        keywords_file_open = open(self.keywords_file, 'r', encoding='utf-8')
        keyword_list = keywords_file_open.readlines()
        id_to_word = {}
        word_to_id = {}
        for keyword in keyword_list:
            id, keyword = keyword.strip().split(' ')
            id_to_word[id] = keyword
            word_to_id[keyword] = id
        keywords_file_open.close()
        word_num = len(word_to_id)
        topic_file_open = open(self.topic_file, 'r', encoding='utf-8')
        topic_string_list = topic_file_open.readlines()
        topic_num = len(topic_string_list)
        print("话题数量:",topic_num,"话题对应关键词数量:", word_num)
        topic_vec = np.zeros([topic_num, word_num])
        for topic_string in topic_string_list:
            topic_string = topic_string.strip()
            topic_id, topic_keywords_string = topic_string.split('---')
            topic_id = int(topic_id)
            topic_id_np_index = topic_id - 1
            keywords_weight_list = topic_keywords_string.split(',')
            for keywords_weight in keywords_weight_list:
                try:
                    keyword, weight = keywords_weight.split(':')
                    keyword = keyword.strip()
                    weight = float(weight.strip())
                    keyword_id = word_to_id[keyword]
                    keyword_id = int(keyword_id)
                    keyword_id_np_index = keyword_id - 1
                    topic_vec[topic_id_np_index, keyword_id_np_index] = weight
                except:
                    print(topic_id, keyword_id, keyword, weight)
        topic_file_open.close()
        np.save(file=self.topic_vec_npy, arr=topic_vec)
        similarity_value = np.inner(topic_vec, topic_vec)
        print(similarity_value.shape)
        np.save(file=self.similar_topic_matrix_npy,arr=similarity_value)

    def get_topic_vec2(self):
        keywords_file_open = open(self.keywords_file, 'r', encoding='utf-8')
        keyword_list = keywords_file_open.readlines()
        id_to_word = {}
        word_to_id = {}
        for keyword in keyword_list:
            id, keyword = keyword.strip().split(' ')
            id_to_word[id] = keyword
            word_to_id[keyword] = id
        keywords_file_open.close()
        word_num = len(word_to_id)
        topic_file_open = open(self.topic_file, 'r', encoding='utf-8')
        topic_string_list = topic_file_open.readlines()
        topic_num = len(topic_string_list)
        print(topic_num, word_num)
        self.topic_num = topic_num
        topic_vec = np.zeros([topic_num, word_num])
        for topic_string in topic_string_list:
            topic_string = topic_string.strip()
            topic_id, topic_keywords_string = topic_string.split('---')
            topic_id = int(topic_id)
            topic_id_np_index = topic_id - 1
            keywords_weight_list = topic_keywords_string.split(',')
            for keywords_weight in keywords_weight_list:
                try:
                    keyword, weight = keywords_weight.split(':')
                    keyword = keyword.strip()
                    weight = float(weight.strip())
                    keyword_id = word_to_id[keyword]
                    keyword_id = int(keyword_id)
                    keyword_id_np_index = keyword_id - 1
                    topic_vec[topic_id_np_index, keyword_id_np_index] = weight
                except:
                    print(topic_id, keyword_id, keyword, weight)
        topic_file_open.close()
        np.save(file=self.topic_vec_npy, arr=topic_vec)
        similarity_value = np.inner(topic_vec, topic_vec)
        # print(similarity_value)
        print("生成话题相似度矩阵:",similarity_value.shape)
        np.save(file=self.similar_topic_matrix_npy,arr=similarity_value)

    def get_similar_topics1(self, target_topic_id):
        target_topic_index = target_topic_id - 1  # topic_id是从1开始存的，而numpy索引是从0开始的
        topic_vec_numpy = np.load(self.topic_vec_npy)
        target_topic_vec = topic_vec_numpy[target_topic_index]
        similarity_value = np.inner(topic_vec_numpy, target_topic_vec)
        similarity_topics = {}
        for index, i in enumerate(similarity_value):
            if float(i) > 0.001 and target_topic_index != index:
                id = index + 1
                similarity_topics[id] = i
        similarity_topics_sorted = sorted(similarity_topics.items(), key = lambda kv:(kv[1], kv[0]))
        # print(similarity_topics_sorted)
        return similarity_topics_sorted

    def get_similar_topics2(self, target_topic_id,threshold=0.001):
        target_topic_index = target_topic_id - 1  # topic_id是从1开始存的，而numpy索引是从0开始的

        target_topic_sim = self.topic_sim_numpy[target_topic_index]
        similarity_topics = {}
        for index, i in enumerate(target_topic_sim):
            if float(i) > threshold and target_topic_index != index:
                id = index + 1
                similarity_topics[id] = i
        similarity_topics_sorted = sorted(similarity_topics.items(), key = lambda kv:(kv[1], kv[0]))
        similarity_topics_sorted = similarity_topics_sorted[::-1]
        return similarity_topics_sorted