"""
常用函数集合
"""

def document2sentences(content):
    """
    将段落或者篇章转为句子集合
    :param content: 篇章内容 []
    :return: []
    """
    try:
        # 结束符号，包含中文和英文的
        end_flag = ['?', '!', '.', '？', '！', '。', '…']
        content_len = len(content)
        sentences = []
        tmp_char = ''
        for idx, char in enumerate(content):
            # 拼接字符
            tmp_char += char

            # 判断是否已经到了最后一位
            if (idx + 1) == content_len:
                sentences.append(tmp_char)
                break

            # 判断此字符是否为结束符号
            if char in end_flag:
                # 再判断下一个字符是否为结束符号，如果不是结束符号，则切分句子
                next_idx = idx + 1
                if not content[next_idx] in end_flag:
                    sentences.append(tmp_char)
                    tmp_char = ''

        return sentences
    except:
        return ''


def check_contain_chinese(check_str):
    """
    判断一个字符串是否 包含 中文字符
    :param check_str: 字符串
    :return: 包含 True 不包含 False
    """
    for ch in check_str.encode('utf-8').decode('utf-8'):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def clean_sentence(sentence):
    """
    清洗单个句子
    :param sentence: 单句字符串
    :return: 清理后的句子字符串
    """
    sentence = sentence.replace('\n', '').replace(' ', '').replace('\t', '').strip()
    return sentence