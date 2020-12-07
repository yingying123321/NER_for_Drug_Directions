#! -*- coding: utf-8 -*-
import time
import os
import tensorflow as tf
import keras
import bert4keras
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm

start = time.time()

maxlen = 250
epochs = 20
batch_size = 16
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

# bert配置
config_path = '/home/liulinhai/llhy/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/liulinhai/llhy/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/liulinhai/llhy/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/vocab.txt'

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                try:
                    char, this_flag = c.split(' ')
                except:
                    print(c)
                    continue
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D

# 标注数据
train_data = load_data('/home/liulinhai/llhy/baseline/data/train.txt')
valid_data = load_data('/home/liulinhai/llhy/baseline/data/val.txt')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射

labels = ['SYMPTOM',
 'DRUG_EFFICACY',
 'PERSON_GROUP',
 'SYNDROME',
 'DRUG_TASTE',
 'DISEASE',
 'DRUG_DOSAGE',
 'DRUG_INGREDIENT',
 'FOOD_GROUP',
 'DISEASE_GROUP',
 'DRUG',
 'FOOD',
 'DRUG_GROUP']

id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

model = build_transformer_model(
    config_path,
    checkpoint_path,
)

def expand2(input):
    return K.expand_dims(input, axis=2)

layer_logits = []
seq_out = []

for i in range(bert_layers):
    output_layer = 'Transformer-%s-FeedForward-Norm' % i
    layer = model.get_layer(output_layer).output
    layer_logits.append(Dense(1,
                        kernel_initializer = keras.initializers.TruncatedNormal(stddev=0.02),
                        name='layer_logit%d' % i)(layer))
    seq_out.append(keras.layers.Lambda(expand2, name='expand_{}'.format(i))(layer))
    
seq_out = keras.layers.Concatenate(axis=2)(seq_out)
layer_logits = keras.layers.Concatenate(axis=2)(layer_logits)
soft = keras.layers.Softmax()
layer_dist = soft(layer_logits)



def matM(inputs):
    x, y = inputs
    x = K.expand_dims(x, axis=2)
    return tf.squeeze(tf.matmul(x, y), axis=2)

output = keras.layers.Lambda(matM, name='matmul')([layer_dist, seq_out])
    

output = Dense(512)(output)

output = keras.layers.Bidirectional(
        keras.layers.LSTM(
            units = 32, return_sequences = True))(output)
output = Dense(32)(output)
output = Dense(num_labels)(output) # 27分类

CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learing_rate),
    metrics=[CRF.sparse_accuracy]
)

class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text)) # 预测
        T = set([tuple(i) for i in d if i[1] != 'O']) #真实
        X += len(R & T) 
        Y += len(R) 
        Z += len(T)
    precision, recall =  X / Y, X / Z
    f1 = 2*precision*recall/(precision+recall)
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    def __init__(self,valid_data):
        self.best_val_f1 = 0
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
#         print(NER.trans)
        f1, precision, recall = evaluate(self.valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model_epoch_10_bertBiLSTMCrf.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

evaluator = Evaluator(valid_data)
train_generator = data_generator(train_data, batch_size)

model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    callbacks=[evaluator]
)

def _cut(sentence):
    """
    将一段文本切分成多个句子
    :param sentence:
    :return:
    """
    new_sentence = []
    sen = []
    for i in sentence:
        if i in ['。', '！', '？', '?'] and len(sen) != 0:
            sen.append(i)
            new_sentence.append("".join(sen))
            sen = []
            continue
        sen.append(i)

    if len(new_sentence) <= 1: # 一句话超过max_seq_length且没有句号的，用","分割，再长的不考虑了。
        new_sentence = []
        sen = []
        for i in sentence:
            if i.split(' ')[0] in ['，', ','] and len(sen) != 0:
                sen.append(i)
                new_sentence.append("".join(sen))
                sen = []
                continue
            sen.append(i)
    if len(sen) > 0:  # 若最后一句话无结尾标点，则加入这句话
        new_sentence.append("".join(sen))
    return new_sentence

def cut_test_set(text_list,len_treshold):
    cut_text_list = []
    cut_index_list = []
    for text in text_list:

        temp_cut_text_list = []
        text_agg = ''
        if len(text) < len_treshold:
            temp_cut_text_list.append(text)
        else:
            sentence_list = _cut(text)  # 一条数据被切分成多句话
            for sentence in sentence_list:
                if len(text_agg) + len(sentence) < len_treshold:
                    text_agg += sentence
                else:
                    temp_cut_text_list.append(text_agg)
                    text_agg = sentence
            temp_cut_text_list.append(text_agg)  # 加上最后一个句子

        cut_index_list.append(len(temp_cut_text_list))
        cut_text_list += temp_cut_text_list

    return cut_text_list, cut_index_list


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        nodes = model.predict([[token_ids], [segment_ids]])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]

NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

def test_predict(data, NER_):
    test_ner =[]
    
    for text in tqdm(data):
        cut_text_list, cut_index_list = cut_test_set([text],maxlen)
        posit = 0
        item_ner = []
        index =1
        for str_ in cut_text_list:
            aaaa  = NER_.recognize(str_)
            for tn in aaaa:
                ans = {}
                ans["label_type"] = tn[1]
                ans['overlap'] = "T" + str(index)
                
                ans["start_pos"] = text.find(tn[0],posit)
                ans["end_pos"] = ans["start_pos"] + len(tn[0])
                posit = ans["end_pos"]
                ans["res"] = tn[0]
                item_ner.append(ans)
                index +=1
        test_ner.append(item_ner)
    
    return test_ner

import glob 
import codecs
X, Y, Z = 1e-10, 1e-10, 1e-10
val_data_flist = glob.glob('/home/liulinhai/llhy/baseline/round1_train/val_data/*.txt')
data_dir = '/home/liulinhai/llhy/baseline/round1_train/val_data/'
for file in val_data_flist:
    if file.find(".ann") == -1 and file.find(".txt") == -1:
        continue
    file_name = file.split('/')[-1].split('.')[0]
    r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
    r_txt_path = os.path.join(data_dir, "%s.txt" % file_name)

    R = []
    with codecs.open(r_txt_path, "r", encoding="utf-8") as f:
        line = f.readlines()
        aa = test_predict(line, NER)
        for line in aa[0]:
            lines = line['label_type']+ " "+str(line['start_pos'])+' ' +str(line['end_pos'])+ "\t" +line['res']
            R.append(lines)    
    T = []
    with codecs.open(r_ann_path, "r", encoding="utf-8") as f:
        for line in f:
            lines = line.strip('\n').split('\t')[1] + '\t' + line.strip('\n').split('\t')[2]
            T.append(lines)
    R = set(R)
    T = set(T)
    X += len(R & T) 
    Y += len(R) 
    Z += len(T)
precision, recall =  X / Y, X / Z
f1 = 2*precision*recall/(precision+recall)
print(f1,precision,recall)
print('The whole process took {} seconds in total'.format(time.time()-start))

