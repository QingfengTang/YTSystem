import pandas as pd
from src.topic.TopicDiscover import TopicDiscover


class YTApp():
    def __init__(self, raw_data):
        # 话题相关
        self.topic = TopicDiscover([raw_data], id_col=0, content_col_name='content', text_num=20000, stop_word_url='./data/stopwords.txt')

    def build(self):
        # 话题分类
        self.topic.build(max_division_depth=10, min_doc_num_per_cate=30, min_nodes_per_cate=30, top_keyword_num=20, setID=True)
        self.topic.toExcel('./save/topic/topic_discovery.xlsx')

    def run(self):
        # 初始化，提前加载数据
        topic_data = pd.read_excel('./save/topic/topic_discovery.xlsx')
        # topic 个数
        num_topic = len(set(topic_data['topic_id']))

        print('共发现了 {} 类话题'.format(num_topic))

        topic_id = input('请输入想要查看的 topic id: ')
        print(topic_id)








if __name__ == '__main__':
    app = YTApp(raw_data='./data/raw_data/America.xlsx')
    # 构建
    # app.build()
    # 展示
    app.run()



