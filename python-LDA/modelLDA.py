import codecs
from gensim import corpora, models, similarities
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np

train = []  # 训练数据
fp = codecs.open(r'output/output.txt', 'r', encoding='utf-8')
for line in fp:
    line = line.split()
    train.append([w for w in line])

dictionary = corpora.Dictionary(train)  # 构造词典
corpus = [dictionary.doc2bow(text) for text in train]  # 每个text对应的稀疏向量
tfidf = models.TfidfModel(corpus)  # 统计tfidf
corpus_tfidf = tfidf[corpus]

# 将文本的tfidf向量输入生成Lsi模型，num_topics为生成主题个数，也为Lsi进行SVD分解，生成矩阵列向量数；id2word是语料字典
lsi = models.LsiModel(corpus_tfidf, num_topics=50, id2word=dictionary)
topic_result = [a for a in lsi[corpus_tfidf]]  # 給lsi的索引为tfidf向量
print(lsi)  # 打印LSI Model topic_result
# print(lsi.print_topics(num_topics=50, num_words=5))   # 打印5个主题并且打印与主题有关的5个关键词，关键词前面的系数为权重而不是概率值

similarity = similarities.MatrixSimilarity(lsi[corpus_tfidf])   # 根据lsi计算文档之间的相似性
# print(list(similarity))

#  alpha，eta即为LDA公式中的α和β，minimum_probability表示主题小于某个值（比如0.001）就舍弃此主题。
lda = models.LdaModel(corpus_tfidf, num_topics=50, id2word=dictionary, alpha='auto', eta='auto', minimum_probability=0.001)

# for doc_topic in lda.get_document_topics(corpus_tfidf):  # 可以获得每个文档的主题分布
#     print(doc_topic)

with open(r'output/wordlistOutput.txt', 'w', encoding='utf-8') as f1:
    for topic_id in range(50):
        print('Topic', topic_id)
        # print(lda.get_topic_terms(topicid=topic_id))  # lda生成的主题中的词分布，默认显示10个
        print(lda.show_topic(topicid=topic_id))
        word_list = lda.show_topic(topicid=topic_id)
        for i in range(10):
            f1.write(word_list[i][0]+'\n')




a = np.array(list(similarity))
result_index = a > 0.99000000
# print(result_index)

inputs = open(r'input/input.txt', 'r', encoding='utf-8')
text_list = inputs.readlines()
# print(text_list)
count = len(text_list)
for line in range(count):
    nowline = count - 1 - line
    num = 0
    for item in result_index[nowline][nowline:count-1]:
        if item and num != 0:
            text_list[nowline] = text_list[nowline+num].replace('\n', '\\n') + text_list[nowline]
            text_list[nowline+num] = ''
        num += 1
# print(text_list)

with open(r'output/finaloutput.txt', 'w', encoding='utf-8') as f:
    for text in text_list:
        f.write(text)


