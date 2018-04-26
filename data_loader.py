import sys
import torch
import torch.autograd as autograd
import codecs
import random
import torch.utils.data as Data

SEED = 1
random.seed(SEED)

# input: a sequence of tokens, and a token_to_index dictionary
# output: a LongTensor variable to encode the sequence of idxs
def prepare_sequence(seq, to_ix, cuda=False):
    var = autograd.Variable(torch.LongTensor([to_ix[w] for w in seq.split(' ')]))
    return var

def prepare_label(label,label_to_ix, cuda=False):
    var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
    return var

def build_token_to_ix(sentences):
    token_to_ix = dict()
    print(len(sentences))
    for sent in sentences:
        for token in sent.split(' '):
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix

def build_label_to_ix(labels):
    label_to_ix = dict()
    for label in labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)


def get_all_files_from_dir(dirpath):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]
    return onlyfiles

def head_n(fname, n=2):
    count = 1
    print '---------------------------------------'
    f = open(fname)
    for line in f:
        print line
        count += 1
        if count > n:
            print '-----------------------------------'
            f.close()
            return
#head_n(test_file_pos,10)
#thos function would only extract vp and vn files, in order to extract vpn = vp+vn in one file, give last arguement as False/0 

def extract_names(l_files,patt,p_xor_n = True):
    # note that you may need to 
    r = []
    for f in l_files:
        tokens = f.split(".")
        if tokens[-3]==patt:
            if p_xor_n:
                if tokens[-2] != 'vpn':
                    r.append(f)
                    #yield f
            elif tokens[-2] == 'vpn':
                return f
    return sorted(r)
#extract_names(files,'test')
def get_sentence_out(path):
    f = open(path)
    return map(lambda x:x.split(",")[2],f)
def get_neg_pos_sent(type_,files,path):
    return [get_sentence_out(path+n) for n in extract_names(files,type_)]
def load_stanford_data():
    fpath = './cross_validation_data/vpn_filtered/'
    files = get_all_files_from_dir(fpath)
    
    train_sent_neg,train_sent_pos = get_neg_pos_sent('train',files,fpath)
    val_sent_neg,val_sent_pos = get_neg_pos_sent('dev',files,fpath)
    test_sent_neg,test_sent_pos = get_neg_pos_sent('test',files,fpath)
    
    train_data = [(sent,1) for sent in train_sent_pos] + [(sent, 0) for sent in train_sent_neg]
    dev_data = [(sent, 1) for sent in val_sent_pos] + [(sent, 0) for sent in val_sent_neg]
    test_data = [(sent, 1) for sent in test_sent_pos] + [(sent, 0) for sent in test_sent_neg]
    
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)

    print('train:',len(train_data),'dev:',len(dev_data),'test:',len(test_data))
    
    word_to_ix = build_token_to_ix([s for s,_ in train_data+dev_data+test_data])
    label_to_ix = {0:0,1:1}
    print('vocab size:',len(word_to_ix),'label size:',len(label_to_ix))
    print('loading data done!')
    return train_data,dev_data,test_data,word_to_ix,label_to_ix