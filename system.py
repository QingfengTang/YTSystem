import pandas as pd
from src.topic.TopicDiscover import TopicDiscover
from src.topic_rec.similar_topic_rec import similar_topic_recommending


class YTApp():
    def __init__(self, raw_data):
        # 话题相关
        self.topic = TopicDiscover([raw_data], id_col=0, content_col_name='content', text_num=20000, stop_word_url='./data/stopwords.txt')
        self.news_topic_file = './save/topic/topic_discovery.xlsx'

        self.keywords_file = './data/topic_rec/keywords.txt'
        self.topic_file = './data/topic_rec/topic_list.txt'
        self.topic_vec_npy = './data/topic_rec/topic_vec.npy'
        self.similar_topic_matrix_npy = './data/topic_rec/similar_topic_matrix.npy'

        self.st = similar_topic_recommending(news_topic_file=self.news_topic_file,
                                        keywords_file=self.keywords_file,
                                        topic_file=self.topic_file,
                                        topic_vec_npy=self.topic_vec_npy,
                                        similar_topic_matrix_npy=self.similar_topic_matrix_npy)

    def build(self):
        # 话题分类
        self.topic.build(max_division_depth=10, min_doc_num_per_cate=30, min_nodes_per_cate=30, top_keyword_num=20, setID=True)
        self.topic.toExcel('./save/topic/topic_discovery.xlsx')

        # 话题相似度
        self.st.sim_topic_rec_build()


    def run(self):
        # 初始化，提前加载数据
        topic_data = pd.read_excel('./save/topic/topic_discovery.xlsx')

        # topic 个数
        num_topic = len(set(topic_data['topic_id']))

        print('共发现了 {} 类话题'.format(num_topic))

        topic_id = int(input('请输入想要查看的 topic id: '))
        # 相似话题推荐
        similar_topic_ids = self.st.sim_topic_rec_run(target_topic_id=topic_id, threshold=0.001)
        print('相关相似话题: {}'.format(similar_topic_ids))








if __name__ == '__main__':
    app = YTApp(raw_data='./data/raw_data/America.xlsx')
    # 构建
    # app.build()
    # 展示
    app.run()



