from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
from nltk.tree import ParentedTree
import sys
import os
import numpy as np
from gensim.models import KeyedVectors
import string
import json
from json import JSONEncoder
import numpy


def drawrst(strtree, fname):
    """ Draw RST tree into a file
    """
    if not fname.endswith(".ps"):
        fname += ".ps"
    cf = CanvasFrame()
    t = Tree.fromstring(strtree)
    tc = TreeWidget(cf.canvas(), t)
    cf.add_widget(tc,10,10) # (10,10) offsets
    cf.print_to_file(fname)
    cf.destroy()


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)



######################################################################
def postOrderTraverse1(root, model, data, ignore):
    if root is not None:
        if isinstance(root, str):
            # words.append(root)
            punc = string.punctuation
            if root not in punc:
                if root in model.wv.vocab:
                    data[root] = model[root]
                else:
                    ignore.add(root)
            return root
        #sub_words = []
        grucell = []
        for i in range(len(root)):
            w = postOrderTraverse1(root[i], model, data, ignore)
            grucell.append(w)
        return grucell


def start_embed(path):
    # Load the pre-trained word2vec model
    print("loading a pretrained model")
    model = KeyedVectors.load_word2vec_format(
        'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz', binary=True)
    print("finish loading the model")
    data = {}
    ignore = set()
    for index in range(2030):
        if index == 0:
            continue
        f_read = open(path+"news_" + str(index) + ".txt", "r", encoding="iso-8859-1")
        print("--------------------------------------------------------")
        print("processing news_" + str(index) + ".txt")
        line_count = 1
        while True:
            str2tree = f_read.readline()  # line
            if str2tree:
                if ":" in str2tree:
                    line = str2tree.split(": ", 1)[1]
                    if line == '\n':
                        print("empty line in news_" + str(index) + ".txt : ", line_count)
                        line_count += 1
                        continue
                    parse_tree = Tree.fromstring(line)
                    newtree = ParentedTree.convert(parse_tree)
                    postOrderTraverse1(newtree[0], model, data, ignore)
                    line_count += 1
                else:
                    continue
            else:
                break
        print("-----------------------------------------------------------")
        f_read.close()

    print("totally embedding words: ", len(data))
    print("totally ignored words: ", len(ignore))

    with open('./embed_words_CFG1.json', 'w') as outfile:
        json.dump(data, outfile, cls=NumpyArrayEncoder)

    res = dict.fromkeys(ignore, 0)
    with open('./ignored_words_CFG1.json', 'w') as outfile:
        json.dump(res, outfile)



def TextClassDataLoader(path, batch_size):
    re_files = np.load(path+'/remove.npy').tolist()
    re_files.append('news_1146.txt')
    with open(path+'/ReCOVery/test.json') as f1:
        test_data = json.load(f1)
        print("len of test: ", len(test_data))
    with open(path+'/ReCOVery/train.json') as f2:
        train_data = json.load(f2)
        print("len of train: ", len(train_data))
    with open(path+'/ReCOVery/eval.json') as f3:
        val_data = json.load(f3)
        print("len of eval: ", len(val_data))
    print("total_number: ", len(test_data) + len(train_data) + len(val_data))

    test_bodytext = []
    test_root = []
    test_label = []
    test_title = []
    for i in range(len(test_data)):
        news_id = test_data[i]['news_id']
        title = test_data[i]['title']
        reliability = test_data[i]['reliability']
        if "news_" + str(news_id+1) + ".txt" in re_files:
            continue
        f_read1 = open(path+"/strtree_RST/news_" + str(news_id+1) + ".txt", "r", encoding="iso-8859-1")
        print("--------------------------------------------------------")
        print("processing test news_" + str(news_id+1) + ".txt")
        str2tree = f_read1.read()
        parse_tree = Tree.fromstring(str2tree)
        news_tree = ParentedTree.convert(parse_tree)
        f_read1.close()

        #getting all cfg
        test_dict={}
        f_read2 = open(path+"/strtree_CGF/news_" + str(news_id+1) + ".txt", "r", encoding="iso-8859-1")
        line_count = 1
        while True:
            str2tree = f_read2.readline()  # line
            if str2tree:
                if ":" in str2tree:
                    line = str2tree.split(": ", 1)[1]
                    if line == '\n':
                        line_count += 1
                        continue
                    parse_tree2 = Tree.fromstring(line)
                    sentence_tree = ParentedTree.convert(parse_tree2)
                    test_dict[line_count] = sentence_tree
                    line_count += 1
                else:
                    continue
            else:
                break
        f_read2.close()
        test_root.append(news_tree)
        test_bodytext.append(test_dict)
        test_label.append(reliability)
        test_title.append(title)


    train_bodytext = []
    train_root = []
    train_label = []
    train_title = []
    for i in range(len(train_data)):
        news_id = train_data[i]['news_id']
        title = train_data[i]['title']
        reliability = train_data[i]['reliability']
        if "news_" + str(news_id+1) + ".txt" in re_files:
            continue

        f_read1 = open(path+"/strtree_RST/news_" + str(news_id+1) + ".txt", "r", encoding="iso-8859-1")
        print("--------------------------------------------------------")
        print("processing training news_" + str(news_id+1) + ".txt")
        str2tree = f_read1.read()
        parse_tree = Tree.fromstring(str2tree)
        news_tree = ParentedTree.convert(parse_tree)
        f_read1.close()


        # getting all cfg
        train_dict = {}
        f_read2 = open(path + "/strtree_CGF/news_" + str(news_id+1) + ".txt", "r", encoding="iso-8859-1")
        line_count = 1
        while True:
            str2tree = f_read2.readline()  # line
            if str2tree:
                if ":" in str2tree:
                    line = str2tree.split(": ", 1)[1]
                    if line == '\n':
                        line_count += 1
                        continue
                    parse_tree2 = Tree.fromstring(line)
                    sentence_tree = ParentedTree.convert(parse_tree2)
                    train_dict[line_count] = sentence_tree
                    line_count += 1
                else:
                    continue
            else:
                break
        f_read2.close()
        train_root.append(news_tree)
        train_bodytext.append(train_dict)
        train_label.append(reliability)
        train_title.append(title)

    batch_train_news = []
    batch_train_sentences=[]
    number_train_group = int(len(train_root) / batch_size)
    for i in range(number_train_group):
        batch_train_news.append(train_root[0+i*batch_size:0+(i+1)*batch_size])
        batch_train_sentences.append(train_bodytext[0 + i * batch_size:0 + (i + 1) * batch_size])



    val_bodytext = []
    val_root = []
    val_label = []
    val_title = []
    for i in range(len(val_data)):
        news_id = val_data[i]['news_id']
        title = val_data[i]['title']
        reliability = val_data[i]['reliability']

        if "news_" + str(news_id+1) + ".txt" in re_files:
            continue

        f_read1 = open(path+"/strtree_RST/news_" + str(news_id+1) + ".txt", "r", encoding="iso-8859-1")
        print("--------------------------------------------------------")
        print("processing val news_" + str(news_id+1) + ".txt")
        str2tree = f_read1.read()
        parse_tree = Tree.fromstring(str2tree)
        news_tree = ParentedTree.convert(parse_tree)
        f_read1.close()
        # getting all cfg
        val_dict = {}
        f_read2 = open(path + "/strtree_CGF/news_" + str(news_id+1) + ".txt", "r", encoding="iso-8859-1")
        line_count = 1
        while True:
            str2tree = f_read2.readline()  # line
            if str2tree:
                if ":" in str2tree:
                    line = str2tree.split(": ", 1)[1]
                    if line == '\n':
                        line_count += 1
                        continue
                    parse_tree2 = Tree.fromstring(line)
                    sentence_tree = ParentedTree.convert(parse_tree2)
                    val_dict[line_count] = sentence_tree
                    line_count += 1
                else:
                    continue
            else:
                break
        f_read2.close()
        val_root.append(news_tree)
        val_bodytext.append(val_dict)
        val_label.append(reliability)
        val_title.append(title)

    batch_val_news = []
    batch_val_sentences = []
    number_val_group = int(len(val_root) / batch_size)
    for i in range(number_val_group):
        batch_val_news.append(val_root[0 + i * batch_size:0 + (i + 1) * batch_size])
        batch_val_sentences.append(val_bodytext[0 + i * batch_size:0 + (i + 1) * batch_size])

    # test[i]/train[i]/val[i] stores all sentences about news i
    # map_test stores root of each news
    # batch_train stores all sentences about batch news
    return test_root,test_bodytext,test_label,test_title,train_root,\
           train_bodytext,train_label,train_title,val_root,val_bodytext,val_label,val_title

if __name__ == '__main__':
    new_data = TextClassDataLoader('./data', batch_size=2)
    print("finish")

