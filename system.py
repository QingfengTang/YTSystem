from src.topic.TopicDiscover import TopicDiscover


class YTApp():
    def __init__(self):
        # 话题相关
        self.topic = TopicDiscover(['./data/raw_data/America.xlsx'], id_col=0, content_col_name='content', text_num=20000, stop_word_url='./data/stopwords.txt')

    def run(self, raw_data):
        # 话题分类
        self.topic.build(max_division_depth=10, min_doc_num_per_cate=30, min_nodes_per_cate=30, top_keyword_num=20, setID=True)
        self.topic.toExcel('./save/topic/topic_discovery.xlsx')



if __name__ == '__main__':
    app = YTApp()
    app.run('./data/raw_data/America.xlsx')


