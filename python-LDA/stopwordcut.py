import jieba


# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())  # 去掉无用空白符
    stopwords = stopwordslist('stopWords/stopwords.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


inputs = open(r'input/input.txt', 'r', encoding='utf-8')
outputs = open(r'output/output.txt', 'w', encoding='utf-8')
for line in inputs:
    line_seg = seg_sentence(line)
    outputs.write(line_seg+'\n')
outputs.close()
inputs.close()