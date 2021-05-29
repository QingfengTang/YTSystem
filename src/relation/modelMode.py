import sys
import os,json,codecs,unicodedata,tqdm
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint,Tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
sys.path.append('..')
from src.utils import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras import backend as K
if(K.backend() == 'tensorflow'):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)


BERT_MAX_LEN = 512
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation


def seq_gather(x):
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])  #一个batch中的句子个数6  返回1维张量(第0个位置)  张量的维度 (0,1,2)
    batch_idxs = K.expand_dims(batch_idxs, 1)  #(6,1)
    idxs = K.concatenate([batch_idxs, idxs], 1)  #(6,2)
    return K.tf.gather_nd(seq, idxs)  #根据idxs索引返回seq相应元素重构tensor  理解：一个句子+tokenid


def E2EModel(bert_config_path, bert_checkpoint_path, num_rels ,save_weights_path):
    bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, seq_len=None)
    for l in bert_model.layers  :  # bert模型微调
        l.trainable = True

    tokens_in = Input(shape=(None,)) # (batchsize,seq_len)
    segments_in = Input(shape=(None,)) # (batchsize,seq_len)
    gold_sub_heads_in = Input(shape=(None,)  )  # 正确主语头token 所有主语实体的头
    gold_sub_tails_in = Input(shape=(None,)  )  # 正确主语尾token 所有主语实体的尾
    sub_head_in = Input(shape=(1,))  # 主语头token 随机选择一个主语实体的头
    sub_tail_in = Input(shape=(1,))  # 主语尾token 随机选择一个主语实体的尾
    gold_obj_heads_in = Input(shape=(None, num_rels))  # 正确宾语头token
    gold_obj_tails_in = Input(shape=(None, num_rels))  # 正确宾语尾token

    tokens, segments, gold_sub_heads, gold_sub_tails, sub_head, sub_tail, gold_obj_heads, gold_obj_tails = tokens_in, segments_in, gold_sub_heads_in, gold_sub_tails_in, sub_head_in, sub_tail_in, gold_obj_heads_in, gold_obj_tails_in
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(tokens)  # 输入句子的mask
    # expand_dims在第二个维度中增加一维，元素个数并不会变|| greater比较第一个参数与第二个参数的大小，返回布尔列表|| cast将参数转化为'float32'类型
    tokens_feature = bert_model([tokens, segments])  # [batch_size, seq_len, bert_dim(768)]

    pred_sub_heads = Dense(1, activation='sigmoid')(tokens_feature)  # 预测主语的头token  全连接神经网络层[batch_size, seq_len, 1]
    pred_sub_tails = Dense(1, activation='sigmoid')(tokens_feature)  # 预测主语的尾token                [batch_size, seq_len, 1]
    # 预测主语的模型
    subject_model = Model([tokens_in, segments_in], [pred_sub_heads, pred_sub_tails])  # 泛型模型 Model[输入],[输出]

    sub_head_feature = Lambda(seq_gather)([tokens_feature, sub_head])  # 理解：(seq + )头tokenid  [batch_size, 1, bert_dim]
    sub_tail_feature = Lambda(seq_gather)([tokens_feature, sub_tail])  # 理解：(seq + )尾tokenid  [batch_size, 1, bert_dim]
    sub_feature = Average()([sub_head_feature, sub_tail_feature])  # 平均值  ->主语特征  [batch_size, 1, bert_dim]

    tokens_feature = Add()([tokens_feature, sub_feature])  # hN + 主语特征  [batch_size, seq_len, bert_dim]
    # 原
    pred_obj_heads = Dense(num_rels, activation='sigmoid')(tokens_feature) # 返回shape为(?,num_rels)  [batch_size, seq_len, rel_num]
    pred_obj_tails = Dense(num_rels, activation='sigmoid')(tokens_feature) # [batch_size, seq_len, rel_num]

    # 预测宾语的模型
    object_model = Model([tokens_in, segments_in, sub_head_in, sub_tail_in], [pred_obj_heads, pred_obj_tails])

    # 预测主语和宾语的整体模型
    hbt_model = Model([tokens_in, segments_in, gold_sub_heads_in, gold_sub_tails_in, sub_head_in, sub_tail_in, gold_obj_heads_in, gold_obj_tails_in],
                      [pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails])

    gold_sub_heads = K.expand_dims(gold_sub_heads, 2)
    gold_sub_tails = K.expand_dims(gold_sub_tails, 2)

    sub_heads_loss = K.binary_crossentropy(gold_sub_heads, pred_sub_heads)
    sub_heads_loss = K.sum(sub_heads_loss * mask) / K.sum(mask)  # loss求均值
    sub_tails_loss = K.binary_crossentropy(gold_sub_tails, pred_sub_tails)
    sub_tails_loss = K.sum(sub_tails_loss * mask) / K.sum(mask)  # loss求均值

    obj_heads_loss = K.sum(K.binary_crossentropy(gold_obj_heads, pred_obj_heads), 2, keepdims=True)
    obj_heads_loss = K.sum(obj_heads_loss * mask) / K.sum(mask)
    obj_tails_loss = K.sum(K.binary_crossentropy(gold_obj_tails, pred_obj_tails), 2, keepdims=True)
    obj_tails_loss = K.sum(obj_tails_loss * mask) / K.sum(mask)

    loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)  # 所有loss相加

    hbt_model.add_loss(loss)

    print("加载模型~")
    subject_model.load_weights(save_weights_path, by_name=True)
    object_model.load_weights(save_weights_path, by_name=True)
    hbt_model.load_weights(save_weights_path, by_name=True)
    hbt_model.compile(optimizer=Adam(1e-5))
    hbt_model.summary()  # 输出模型各层的参数状况
    return subject_model, object_model, hbt_model

def extract_items(subject_model, object_model, tokenizer, text_in, id2rel,sub_bar=0.5, obj_bar=0.5):  #返回预测的三元组 h_bar应该是头实体的阈值,同理t_bar是尾实体的阈值，设置多少最好
    tokens = tokenizer.tokenize(text_in)
    token_ids, segment_ids = tokenizer.encode(first=text_in)
    token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])
    if len(token_ids[0]) > BERT_MAX_LEN:
        token_ids = token_ids[:,:BERT_MAX_LEN]
        segment_ids = segment_ids[:,:BERT_MAX_LEN]
    sub_heads_logits, sub_tails_logits = subject_model.predict([token_ids, segment_ids])  #[1,seqlen,1]
    sub_heads, sub_tails = np.where(sub_heads_logits[0] > sub_bar)[0], np.where(sub_tails_logits[0] > sub_bar)[0]
    #np.where(condition)只有condition没有x和y，则输出满足条件 (即非0) 的元素的位置，由于sub_heads_logits[0]为2维，因此np.where返回的是2位，分别是满足要求的元素的第0维位置和第一维位置
    subjects = []
    for sub_head in sub_heads:
        sub_tail = sub_tails[sub_tails >= sub_head]
        if len(sub_tail) > 0:
            sub_tail = sub_tail[0] #选满足条件的最小的（第一个）
            subject = tokens[sub_head: sub_tail+1] #if isChinese else tokens[sub_head: sub_tail+1]  #由于中文没有空格，取字符加1 ,NYT数据这里跟中文的处理结果一样
            subjects.append((subject, sub_head, sub_tail))
    if subjects:
        triple_list = []
        token_ids = np.repeat(token_ids, len(subjects), 0) #之前token_ids是[1,seq_len],现在token_ids[len(subjects),seq_len]
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        sub_heads, sub_tails = np.array([sub[1:] for sub in subjects]).T.reshape((2, -1, 1)) #原先(len(subjects),2) ，变为(2,len(subjects),1)
        obj_heads_logits, obj_tails_logits = object_model.predict([token_ids, segment_ids,sub_heads, sub_tails])  #根据预测的主语头尾，分别预测宾语  (n个主语,句长,2个类别)
        # obj_heads_logits, obj_tails_logits   shape为(1,seq.len,2)
        for i, subject in enumerate(subjects):
            sub = subject[0]
            sub = ''.join([i.lstrip("##") for i in sub]) #lstrip截掉字符
            sub = ' '.join(sub.split('[unused1]'))
            obj_heads, obj_tails = np.where(obj_heads_logits[i] > obj_bar), np.where(obj_tails_logits[i] > obj_bar)

            for obj_head, rel_head in zip(*obj_heads): #解压缩
                for obj_tail, rel_tail in zip(*obj_tails): #解压缩
                    if obj_head <= obj_tail and rel_head == rel_tail: #关系的类型应该一致， 头token 和 尾token应该在同一类关系中，即同一列
                        rel = id2rel[rel_head]
                        obj = tokens[obj_head: obj_tail+1 ] #if isChinese else tokens[obj_head: obj_tail] #由于中文没有空格，取字符加1 ，NYT同上
                        obj = ''.join([i.lstrip("##") for i in obj])
                        obj = ' '.join(obj.split('[unused1]'))
                        triple_list.append((sub, rel, obj))
                        break  #一个主语可能有多个关系和宾语三元组 即同一个主语可能有多个三元组
        triple_set = set()
        for s, r, o in triple_list:
            triple_set.add((s, r, o))
        return list(triple_set)
    else:
        return []

#写入excel文件
def metric2excel(subject_model, object_model, eval_data, id2rel, tokenizer, content,):
    data = pd.DataFrame()
    for line in eval_data:
        Pred_triples = set(extract_items(subject_model, object_model, tokenizer, line, id2rel))
        if Pred_triples:
            for triple in Pred_triples:
                newline = {}
                newline['entity1'] = triple[0]
                newline['relation'] = triple[1]
                newline['entity2'] = triple[2]
                newline['line'] = line
                newline['content'] = content
                newdata = pd.DataFrame.from_dict(newline,orient='index').T
                data = data.append(newdata,ignore_index=True)
                # newdata.to_excel(output_path,encoding='utf-8',index=False,header=True)
    return data

class HBTokenizer(Tokenizer):#复写tokenizer._tokenize()和_word_piece_tokenize()
    def __init__(self,token_dict, cased=True, token_cls = TOKEN_CLS, token_sep = TOKEN_SEP, token_unk = TOKEN_UNK, pad_index = 0):
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        self._token_cls = token_cls
        self._token_sep = token_sep
        self._token_unk = token_unk
        self._pad_index = pad_index
        self._cased = cased

    def _tokenize(self, text):
        if not self._cased:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        spaced = ''
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch): #理解text去除非法字符？
                continue
            else:
                spaced += ch
        tokens = []
        for word in list(spaced):#按空格分token
            tokens += self._word_piece_tokenize(word)
        return tokens

def get_tokenizer(vocab_path): #获取字典

    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return HBTokenizer(token_dict, cased=True)

def loadModel():

    save_weights_path = os.path.join(config.BASE_DIR, 'models\\relation', 'best_model.weights')
    bert_config_path = os.path.join(config.BASE_DIR, 'models\\relation',
                                    'pretrained_bert_models\\chinese_L-12_H-768_A-12\\bert_config.json')
    bert_vocab_path = os.path.join(config.BASE_DIR, 'models\\relation',
                                   'pretrained_bert_models\\chinese_L-12_H-768_A-12\\vocab.txt')
    bert_checkpoint_path = os.path.join(config.BASE_DIR, 'models\\relation',
                                        'pretrained_bert_models\\chinese_L-12_H-768_A-12\\bert_model.ckpt')
    rel_dict_path = os.path.join(config.BASE_DIR, 'models\\relation', 'rel2id.json')
    id2rel, rel2id = json.load(open(rel_dict_path, encoding='utf-8'))
    id2rel = {int(i): j for i, j in id2rel.items()}
    num_rels = len(id2rel)

    subject_model, object_model, hbt_model = E2EModel(bert_config_path, bert_checkpoint_path, num_rels,
                                                      save_weights_path)
    hbt_model.load_weights(save_weights_path)

    tokenizer = get_tokenizer(bert_vocab_path)
    return subject_model,object_model,id2rel,tokenizer


