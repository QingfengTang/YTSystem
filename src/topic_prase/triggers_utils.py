import os
import re
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser
import json
import sys
from src.utils.config import BASE_DIR
LTP_DATA_DIR = os.path.join(BASE_DIR,'models\\topic_prase\\ltp_data_v3.4.0')
segmentor_DIR = os.path.join(LTP_DATA_DIR,'cws.model')
segmentor = Segmentor()
segmentor.load(segmentor_DIR)  # 分词模型路径，模型名称为`cws.model`

postagger = Postagger()
postagger.load(os.path.join(LTP_DATA_DIR, 'pos.model'))  # 加载词性标注模型

parser = Parser()  # 初始化实例
parser.load(os.path.join(LTP_DATA_DIR, 'parser.model'))  # 依存句法分析模型路径，模型名称为`parser.model`

def parser_main(sentence):
    # words_list = []
    # postags_list = []
    # for sentence in sentences:
    #     words = list(segmentor.segment(sentence))
    #     postags = postagger.postag(words)
    #     words_list.append(words)
    #     postags_list.append(postags)


    words = list(segmentor.segment(sentence))  # 分词
    postags = list(postagger.postag(words))  # 词性标注
    # print(postags)
    return words, postags


def extract_nvn(sentence):
    mccx = ['j', 'nh', 'ni', 'nl', 'nz', 'n']
    # words, postags, child_dict_list, roles_dict, arcs = parser_main(sentence)
    words, postags = parser_main(sentence)
    nvn_list = []
    verb_list = []
    for i in range(len(postags)):
        sz = i
        if postags[i] == 'v' and len(words[i]) > 1:
            verb_list.append(words[i])
            # print(words[i])
            if postags[sz - 1] in mccx and len(words[sz - 1]) > 1:
                nvn_list.append(words[sz - 1] + words[sz])
            if sz < len(postags) - 1:
                if postags[sz + 1] in mccx and len(words[sz + 1]) > 1:
                    nvn_list.append(words[sz] + words[sz + 1])

    # need_list = list(set(nvn_list))
    need_data = []
    if len(nvn_list):
        need_data = nvn_list
    else:
        need_data = []
    verb_str = ''
    for i in verb_list:
        verb_str += i + ' '
    verb_str = verb_str.strip()

    # for i in range(len(postags)):
    #     print(words[i]+ '==='+ postags[i])
    # need_data才是有用的
    # return need_data, verb_str
    return need_data


def extract_db(sentence):
    # words, postags, child_dict_list, roles_dict, arcs = parser_main(sentence)
    words, postags = parser_main(sentence)
    arcs = parser.parse(words, postags)  # 句法分析
    arc_list = []
    for arc in arcs:
        arc_list.append((arc.head, arc.relation))

    db = []
    for j in arc_list:
        if j[1] == 'VOB':
            dong = words[j[0] - 1]
            # print(dong)
            bing = words[arc_list.index(j)]
            if len(bing) > 1 and len(dong) > 1:
                db.append(dong + bing)
    # if db == ' ' or db == '':
    # if len(db) == 0:
    #     db = '无'
    # print('==》' + db)
    return db


def extract_xvx(sentence):
    mccx = ['j', 'nh', 'ni', 'nl', 'nz', 'n', 'nt', 'ns', 'nd']
    # words, postags, child_dict_list, roles_dict, arcs = parser_main(sentence)
    words, postags = parser_main(sentence)
    # print('words==>', words)
    # print('postags==>', postags)
    len_list = []
    for word in words:
        word_len = len(word)
        len_list.append(word_len)
    # print('len_list==>', len_list)
    xvx_list = []
    for i in range(len(postags)):
        sz = i
        # print('sz==>',sz)
        if postags[sz] == 'v' and len(words[sz]) > 1:
            if sz != 0 and sz != len(postags) - 1:
                # if len_list[sz - 1] > 1 and len_list[sz + 1] > 1:
                #     xvx_list.append(words[sz - 1] + words[sz] + words[sz + 1])
                id_r = sz + 1
                while len_list[id_r] <= 1 and id_r < len(postags) - 1:
                    id_r += 1
                real_idr = id_r
                if real_idr == len(postags):
                    real_idr -= 1

                id_l = sz - 1
                while len_list[id_l] <= 1 and id_l >= 0:
                    id_l -= 1
                real_idl = id_l
                if real_idl == -1:
                    real_idl = 0
                data = ''
                jilu_real_idl = real_idl
                jilu_real_idr = real_idr
                # print('jilu_real_idr', jilu_real_idr)
                # print('jilu_real_idl', jilu_real_idl)
                for j in range(real_idl, real_idr + 1):
                    # print(words[real_idl] + '====' + words[real_idr])
                    # print(real_idl,words[real_idr])
                    data += words[j]

                # 补左边名词
                while real_idl != -1 and postags[real_idl] in mccx:
                    real_idl -= 1
                # print(real_idl)
                final_idl = real_idl + 1
                # print('===>final_idl',final_idl)
                lbu_data = ''
                if final_idl == jilu_real_idl:
                    pass
                else:
                    for k in range(final_idl, jilu_real_idl):
                        lbu_data += words[k]
                data = lbu_data + data
                # 补左边名词结束

                # 补右边名词
                # print('+前real_idr', real_idr)
                while real_idr != len(postags) and postags[real_idr] in mccx:
                    # print('我进来啦')
                    real_idr += 1
                # print('+后real_idr', real_idr)
                final_idr = real_idr - 1
                rbu_data = ''
                if real_idr == jilu_real_idr:
                    # print('啥也没干')
                    pass
                elif real_idr == -1:
                    pass
                elif final_idr == jilu_real_idr:
                    pass
                elif jilu_real_idr == len(postags) - 1:
                    pass
                elif (final_idr + 1) < len(postags):
                    # if (final_idr + 1) != len(postags):
                    for k in range(jilu_real_idr + 1, final_idr + 1):
                        rbu_data += words[k]
                # print('rbu_data',rbu_data)
                data = data + rbu_data
                # 补右边名词结束

                xvx_list.append(data)
            elif sz == 0:
                if len(postags) > 1:
                    id = sz + 1
                    while len_list[id] <= 1 and id < len(postags) - 1:
                        id += 1
                    real_id = id
                    jilu_real_id = real_id
                    if real_id == len(postags):
                        data = sentence
                    else:
                        data = ''
                        for j in range(real_id + 1):
                            data += words[j]

                        # 补右边名词
                        # print('+前real_idr', real_id)
                        while real_id != len(postags) and postags[real_id] in mccx:
                            # print('我进来啦')
                            real_id += 1
                        # print('+后real_idr', real_id)
                        final_idr = real_id - 1
                        rbu_data = ''
                        if real_id == jilu_real_id:
                            # print('啥也没干')
                            pass
                        elif real_id == -1:
                            pass
                        elif final_idr == jilu_real_id:
                            pass
                        elif jilu_real_id == len(postags) - 1:
                            pass
                        elif (final_idr + 1) < len(postags):
                            # if (final_idr + 1) != len(postags):
                            for k in range(jilu_real_id + 1, final_idr + 1):
                                rbu_data += words[k]
                        # print('rbu_data',rbu_data)
                        data = data + rbu_data
                        # 补右边名词结束

                    xvx_list.append(data)
                else:
                    data = words[0]
                    xvx_list.append(data)
            elif sz == len(postags) - 1:
                id = sz - 1
                while len_list[id] <= 1 and id >= 0:
                    id -= 1
                real_idl = id
                jilu_real_idl = real_idl
                if real_idl == -1:
                    data = sentence
                else:
                    data = ''
                    for j in range(real_idl, len(postags)):
                        data += words[j]
                    # 补左边名词
                    while real_idl != -1 and postags[real_idl] in mccx:
                        real_idl -= 1
                    # print(real_idl)
                    final_idl = real_idl + 1
                    # print('===>final_idl', final_idl)
                    lbu_data = ''
                    if final_idl == jilu_real_idl:
                        pass
                    else:
                        for k in range(final_idl, jilu_real_idl):
                            lbu_data += words[k]
                    data = lbu_data + data
                    # 补左边名词结束
                xvx_list.append(data)
            # break
            # print('=====')
            # print('=====')
            # print('=====')

    # print('xvx_list==>', xvx_list)

    if len(xvx_list):
        need_data = ''
        for p in xvx_list:
            need_data += p + ' '
        need_data = need_data.strip()
        # print(need_data)
        return need_data
    else:
        need_data = '无'
        return need_data


def format_labelrole(words, postags):
    arcs = parser.parse(words, postags)
    roles = labeller.label(words, postags, arcs)
    roles_dict = {}
    for role in roles:
        roles_dict[role.index] = {arg.name: [arg.name, arg.range.start, arg.range.end] for arg in role.arguments}
    return roles_dict


# 句法分析---为句子中的每个词语维护一个保存句法依存儿子节点的字典
def build_parse_child_dict(words, postags, arcs):
    child_dict_list = []
    format_parse_list = []
    for index in range(len(words)):
        child_dict = dict()
        for arc_index in range(len(arcs)):
            if arcs[arc_index].head == index + 1:  # arcs的索引从1开始，找到谓语核心词
                if arcs[arc_index].relation in child_dict:
                    child_dict[arcs[arc_index].relation].append(arc_index)
                else:
                    child_dict[arcs[arc_index].relation] = []
                    child_dict[arcs[arc_index].relation].append(arc_index)
        child_dict_list.append(child_dict)
    rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
    relation = [arc.relation for arc in arcs]  # 提取依存关系
    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
    for i in range(len(words)):
        # ['ATT', '李克强', 0, 'nh', '总理', 1, 'n']
        a = [relation[i], words[i], i, postags[i], heads[i], rely_id[i] - 1, postags[rely_id[i] - 1]]
        format_parse_list.append(a)

    return child_dict_list, format_parse_list


def complete_e(words, postags, child_dict_list, word_index):
    child_dict = child_dict_list[word_index]
    prefix = ''
    if 'ATT' in child_dict:
        for i in range(len(child_dict['ATT'])):
            prefix += complete_e(words, postags, child_dict_list, child_dict['ATT'][i])
    postfix = ''
    if postags[word_index] == 'v':
        if 'VOB' in child_dict:
            postfix += complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
        if 'SBV' in child_dict:
            prefix = complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

    return prefix + words[word_index] + postfix


def ruler1(words, postags, roles_dict, role_index):
    # 利用语义角色标注,直接获取主谓宾三元组,基于A0,A1,A2
    v = words[role_index]  # 找到谓语角色词
    role_info = roles_dict[role_index]  # 找到谓语角色词的元祖
    # print(words)
    if 'A0' in role_info.keys() and 'A1' in role_info.keys():
        # [words[word_index] for word_index in range(role_info['A0'][1], role_info['A0'][2]+1) if postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]]
        # print(words[word_index])
        # [A0]代表主语，提取主语
        s = ''.join([words[word_index] for word_index in range(role_info['A0'][1], role_info['A0'][2] + 1) if
                     postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])  # u	auxiliary||x	non-lexeme
        o = ''.join([words[word_index] for word_index in range(role_info['A1'][1], role_info['A1'][2] + 1) if
                     postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
        # print(s)
        # print(o)
        if s and o:
            return '1', [s, v, o]
    return '4', []


def ruler2(words, postags, child_dict_list, arcs, roles_dict):
    svos = []
    for index in range(len(postags)):
        tmp = 1
        # 先借助语义角色标注的结果，进行三元组抽取
        if index in roles_dict:  # 说明是谓词的语义角色
            flag, triple = ruler1(words, postags, roles_dict, index)
            if flag == '1':
                svos.append(triple)  # ruler提取成功，svo都已经提取
                tmp = 0
        if tmp == 1:  # 经过词性标注分析，ruler提取不出svo
            # 如果语义角色标记为空，则使用依存句法进行抽取
            # if postags[index] == 'v':
            if postags[index]:
                # 抽取以谓词为中心的事实三元组
                child_dict = child_dict_list[index]
                # 主谓宾
                if 'SBV' in child_dict and 'VOB' in child_dict:  # 子节点有主语和宾语，说明是谓语
                    r = words[index]  # 抽取谓语的词
                    e1 = complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                    # print(child_dict['SBV'][0]+"123")
                    # print(child_dict['VOB'][0]+"456")
                    e2 = complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                    svos.append([e1, r, e2])

                # 定语后置，动宾关系
                relation = arcs[index][0]
                head = arcs[index][2]
                # print(arcs[index])
                if relation == 'ATT':
                    if 'VOB' in child_dict:
                        e1 = complete_e(words, postags, child_dict_list, head - 1)
                        r = words[index]
                        e2 = complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                        temp_string = r + e2
                        if temp_string == e1[:len(temp_string)]:
                            e1 = e1[len(temp_string):]
                        if temp_string not in e1:
                            svos.append([e1, r, e2])
                # 含有介宾关系的主谓动补关系
                if 'SBV' in child_dict and 'CMP' in child_dict:
                    e1 = complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                    cmp_index = child_dict['CMP'][0]
                    r = words[index] + words[cmp_index]
                    if 'POB' in child_dict_list[cmp_index]:
                        e2 = complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
                        svos.append([e1, r, e2])
    return svos


def triples_main(content):
    sentences = split_sents(content)
    svos = []
    for sentence in sentences:
        words, postags, child_dict_list, roles_dict, arcs = parser_main(sentence)
        svo = ruler2(words, postags, child_dict_list, arcs, roles_dict)
        svos += svo

    return svos


def split_sents(content):
    # 文章分句处理, 切分长句，冒号，分号，感叹号等做切分标识
    return [sentence for sentence in re.split(r'[,，？?！!。；;：:\n\r]', content)]


# def test(content):
#     # 测试
#     svos = triples_main(content)
#     return svos




if __name__ == '__main__':
    # print(time.localtime())
    LTP_DATA_DIR = './ltp_data_v3.4.0'  # ltp模型目录的路径

    segmentor = Segmentor()
    cwspath = r'C:\Users\xuchanghua\PycharmProjects\YTCodebase\src\topic_parse\model\ltp_data_v3.4.0\cws.model'
    # segmentor.load(os.path.join(LTP_DATA_DIR, 'cws.model'))  # 分词模型路径，模型名称为`cws.model`
    segmentor.load(cwspath)

    postagger = Postagger()
    postagger.load(os.path.join(LTP_DATA_DIR, 'pos.model'))  # 加载词性标注模型

    parser = Parser()  # 初始化实例
    parser.load(os.path.join(LTP_DATA_DIR, 'parser.model'))  # 依存句法分析模型路径，模型名称为`parser.model`
    sentence = "特朗普是个优秀的分词工具"
    words = segmentor.segment(sentence)
    words_list = list(words)
    print(words_list)
    """

    def segmentor(sentence):
        segmentor = Segmentor()  # 初始化实例
        segmentor.load(cws_model_path)  # 加载模型
        words = segmentor.segment(sentence)  # 分词
        # 默认可以这样输出
        # 可以转换成List 输出
        words_list = list(words)
        segmentor.release()  # 释放模型
        return words_list
    """

# 加载数据
    data = json.load(open('./splited2sents.json', encoding='utf-8'))

    # triggers extract
    result = {}
    for k, v in data.items():
        triggers = []
        for i in range(len(v)):
            triggers.append([extract_xvx(v[i]), extract_nvn(v[i]), extract_db(v[i])])

        result[k] = triggers
        print('processing: ', str(int(k) / 11505 * 100))

    triggers2json = json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False)
    with open('./alltriggers.json', 'w', encoding='utf-8') as f:
        f.write(triggers2json)
    print('ok')

