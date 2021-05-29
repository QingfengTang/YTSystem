from src.topic.TopicDiscover import TopicDiscover

if __name__ == '__main__':
    td = TopicDiscover(['../data/raw_data/America.xlsx'], id_col=0, content_col_name='content', text_num=20000, stop_word_url='../data/stopwords.txt')
    # split_by = td.get_split_tag('sentence')
    # td.split_data(split_by)
    td.build(max_division_depth=200, min_doc_num_per_cate=20, min_nodes_per_cate=10, top_keyword_num=20, setID=True)
    # td.delete_NoTopic()
    td.toExcel('../save/topic/topic_discovery1.xlsx')