from src.topic_rec.similar_topic_rec import similar_topic_recommending

if __name__ == '__main__':
    news_topic_file = '../save/topic/topic_discovery.xlsx'

    keywords_file = '../data/topic_rec/keywords.txt'
    topic_file = '../data/topic_rec/topic_list.txt'
    topic_vec_npy = '../data/topic_rec/topic_vec.npy'
    similar_topic_matrix_npy = '../data/topic_rec/similar_topic_matrix.npy'

    st = similar_topic_recommending(news_topic_file=news_topic_file,
                                    keywords_file=keywords_file,
                                    topic_file=topic_file,
                                    topic_vec_npy=topic_vec_npy,
                                    similar_topic_matrix_npy=similar_topic_matrix_npy)
    #### 如果之前已经运行过，生成过相关文件，且没更新话题，build可以注释
    st.sim_topic_rec_build()

    #### 正常情况下，话题id是从1开始且连续的
    topic_num = st.get_topic_num()
    print("话题数量:",topic_num)

    result = st.sim_topic_rec_run(target_topic_id=8,threshold=0.001)
    print(result)

    result = st.sim_topic_rec_run(target_topic_id=70,threshold=0.001)
    print(result)