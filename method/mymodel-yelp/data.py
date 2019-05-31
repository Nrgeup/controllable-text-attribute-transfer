import numpy as np
import os
import random
import torch
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction
import nltk


def calc_bleu(reference, hypothesis):
    weights = (0.25, 0.25, 0.25, 0.25)
    return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights,
                                                   smoothing_function=SmoothingFunction().method1)


def load_human_answer(data_path):
    ans = []
    file_list = [
        data_path + 'reference.0',
        data_path + 'reference.1',
    ]
    for file in file_list:
        with open(file) as f:
            for line in f:
                line = line.strip()
                line = line.split('\t')[1].split()
                parse_line = [int(x) for x in line]
                ans.append(parse_line)
    return ans


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def id2text_sentence(sen_id, id_to_word):
    sen_text = []
    max_i = len(id_to_word)
    for i in sen_id:
        if i == 3:  # id_eos
            break
        if i >= max_i:
            i = 1  # UNK
        sen_text.append(id_to_word[i])
    return ' '.join(sen_text)


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def get_cuda(tensor):
    # if torch.cuda.is_available():
    #     tensor = tensor
    return tensor.cuda()


def load_word_dict_info(word_dict_file, max_num):
    id_to_word = []
    with open(word_dict_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip()
            item_list = item.split('\t')
            word = item_list[0]
            if len(item_list) > 1:
                num = int(item_list[1])
                if num < max_num:
                    break
            id_to_word.append(word)
    print("Load word-dict with %d size and %d max_num." % (len(id_to_word), max_num))
    return id_to_word, len(id_to_word)


def load_data1(file1):
    token_stream = []
    with open(file1, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            token_stream.append(parse_line)
    return token_stream


def prepare_data(data_path, max_num, task_type):
    print("prepare data ...")
    id_to_word, vocab_size = load_word_dict_info(data_path + 'word_to_id.txt', max_num)

    # define train / test file
    train_file_list = []
    train_label_list = []
    if task_type == 'yelp' or task_type == 'amazon':
        train_file_list = [
            data_path + 'sentiment.train.0', data_path + 'sentiment.train.1',
            data_path + 'sentiment.dev.0', data_path + 'sentiment.dev.1',
        ]
        train_label_list = [
            [0],
            [1],
            [0],
            [1],
        ]

    return id_to_word, vocab_size, train_file_list, train_label_list


def pad_batch_seuqences(origin_seq, sos_id, eos_id, unk_id, max_seq_length, vocab_size):
    '''padding with 0, mask id_num > vocab_size with unk_id.'''
    max_l = 0
    for i in origin_seq:
        max_l = max(max_l, len(i))

    max_l = min(max_seq_length, max_l + 1)

    encoder_input_seq = np.zeros((len(origin_seq), max_l-1), dtype=int)
    decoder_input_seq = np.zeros((len(origin_seq), max_l), dtype=int)
    decoder_target_seq = np.zeros((len(origin_seq), max_l), dtype=int)
    encoder_input_seq_length = np.zeros((len(origin_seq)), dtype=int)
    decoder_input_seq_length = np.zeros((len(origin_seq)), dtype=int)
    for i in range(len(origin_seq)):
        decoder_input_seq[i][0] = sos_id
        for j in range(min(max_l-1, len(origin_seq[i]))):
            this_id = origin_seq[i][j]
            if this_id >= vocab_size:
                this_id = unk_id
            encoder_input_seq[i][j] = this_id
            decoder_input_seq[i][j + 1] = this_id
            decoder_target_seq[i][j] = this_id
        encoder_input_seq_length[i] = min(max_l-1, len(origin_seq[i]))
        decoder_input_seq_length[i] = min(max_l, len(origin_seq[i]) + 1)
        decoder_target_seq[i][decoder_input_seq_length[i]-1] = eos_id
    return encoder_input_seq, decoder_input_seq, decoder_target_seq, encoder_input_seq_length, decoder_input_seq_length


class non_pair_data_loader():
    def __init__(self, batch_size, id_bos, id_eos, id_unk, max_sequence_length, vocab_size):
        self.sentences_batches = []
        self.labels_batches = []

        self.src_batches = []
        self.src_mask_batches = []
        self.tgt_batches = []
        self.tgt_y_batches = []
        self.tgt_mask_batches = []
        self.ntokens_batches = []

        self.num_batch = 0
        self.batch_size = batch_size
        self.pointer = 0
        self.id_bos = id_bos
        self.id_eos = id_eos
        self.id_unk = id_unk
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size


    def create_batches(self, train_file_list, train_label_list, if_shuffle=True):
        self.data_label_pairs = []
        for _index in range(len(train_file_list)):
            with open(train_file_list[_index]) as fin:
                for line in fin:
                    line = line.strip()
                    line = line.split()
                    parse_line = [int(x) for x in line]
                    self.data_label_pairs.append([parse_line, train_label_list[_index]])

        if if_shuffle:
            random.shuffle(self.data_label_pairs)

        # Split batches
        if self.batch_size == None:
            self.batch_size = len(self.data_label_pairs)
        self.num_batch = int(len(self.data_label_pairs) / self.batch_size)
        for _index in range(self.num_batch):
            item_data_label_pairs = self.data_label_pairs[_index*self.batch_size:(_index+1)*self.batch_size]
            item_sentences = [_i[0] for _i in item_data_label_pairs]
            item_labels = [_i[1] for _i in item_data_label_pairs]

            batch_encoder_input, batch_decoder_input, batch_decoder_target, \
            batch_encoder_length, batch_decoder_length = pad_batch_seuqences(
                item_sentences, self.id_bos, self.id_eos, self.id_unk, self.max_sequence_length, self.vocab_size,)

            src = get_cuda(torch.tensor(batch_encoder_input, dtype=torch.long))
            tgt = get_cuda(torch.tensor(batch_decoder_input, dtype=torch.long))
            tgt_y = get_cuda(torch.tensor(batch_decoder_target, dtype=torch.long))

            src_mask = (src != 0).unsqueeze(-2)
            tgt_mask = self.make_std_mask(tgt, 0)
            ntokens = (tgt_y != 0).data.sum().float()

            # For debug
            # print("item_sentences", item_sentences)
            # print("item_labels", item_labels)
            # print("src", src)
            # print("tgt", tgt)
            # print("tgt_y", tgt_y)
            # print("batch_encoder_length", batch_encoder_length)
            # print("batch_decoder_length", batch_decoder_length)
            # print("src_mask", src_mask)
            # print("tgt_mask", tgt_mask)
            # print("ntokens", ntokens.float())
            # input("--------------")

            self.sentences_batches.append(item_sentences)
            self.labels_batches.append(get_cuda(torch.tensor(item_labels, dtype=torch.float)))
            self.src_batches.append(src)
            self.tgt_batches.append(tgt)
            self.tgt_y_batches.append(tgt_y)
            self.src_mask_batches.append(src_mask)
            self.tgt_mask_batches.append(tgt_mask)
            self.ntokens_batches.append(ntokens)

        self.pointer = 0
        print("Load data from %s !\nCreate %d batches with %d batch_size" % (
            ' '.join(train_file_list), self.num_batch, self.batch_size
        ))

    def make_std_mask(self, tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


    def next_batch(self):
        """take next batch by self.pointer"""
        this_batch_sentences = self.sentences_batches[self.pointer]
        this_batch_labels = self.labels_batches[self.pointer]

        this_src = self.src_batches[self.pointer]
        this_src_mask = self.src_mask_batches[self.pointer]
        this_tgt = self.tgt_batches[self.pointer]
        this_tgt_y = self.tgt_y_batches[self.pointer]
        this_tgt_mask = self.tgt_mask_batches[self.pointer]
        this_ntokens = self.ntokens_batches[self.pointer]

        self.pointer = (self.pointer + 1) % self.num_batch
        return this_batch_sentences, this_batch_labels, \
               this_src, this_src_mask, this_tgt, this_tgt_y, \
               this_tgt_mask, this_ntokens


    def reset_pointer(self):
        self.pointer = 0


if __name__ == '__main__':



    class Batch:
        "Object for holding a batch of data with mask during training."

        def __init__(self, src, trg=None, pad=0):
            self.src = src
            self.src_mask = (src != pad).unsqueeze(-2)
            if trg is not None:
                self.trg = trg[:, :-1]
                self.trg_y = trg[:, 1:]
                self.trg_mask = \
                    self.make_std_mask(self.trg, pad)
                self.ntokens = (self.trg_y != pad).data.sum()

        @staticmethod
        def make_std_mask(tgt, pad):
            "Create a mask to hide padding and future words."
            tgt_mask = (tgt != pad).unsqueeze(-2)
            tgt_mask = tgt_mask & Variable(
                subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
            return tgt_mask


    def data_gen(V, batch, nbatches):
        "Generate random data for a src-tgt copy task."
        for i in range(nbatches):
            data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
            data[:, 0] = 1
            src = Variable(data, requires_grad=False)
            tgt = Variable(data, requires_grad=False)
            yield Batch(src, tgt, 0)


    for i in range(100):
        print("%d ----- " % i)
        data_iter = data_gen(10, 3, 2)
        for j, batch in enumerate(data_iter):
            print("%d:", j)
            print(batch.src)
            print(batch.src_mask)
            print(batch.trg)
            print(batch.trg_y)
            print(batch.trg_mask)
            input("=====")



