# coding: utf-8
# requirements: pytorch: 0.4
# Author: Ke Wang
# Contact: wangke17[AT]pku.edu.cn
import time
import argparse
import math
import os
import torch
import torch.nn as nn
from torch import optim
import numpy
import matplotlib
from matplotlib import pyplot as plt

# Import your model files.
from model import make_model, Classifier, NoamOpt, LabelSmoothing, fgim_attack
from data import prepare_data, non_pair_data_loader, get_cuda, pad_batch_seuqences, id2text_sentence,\
    to_var, calc_bleu, load_human_answer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

######################################################################################
#  Environmental parameters
######################################################################################
parser = argparse.ArgumentParser(description="Here is your model discription.")
parser.add_argument('--id_pad', type=int, default=0, help='')
parser.add_argument('--id_unk', type=int, default=1, help='')
parser.add_argument('--id_bos', type=int, default=2, help='')
parser.add_argument('--id_eos', type=int, default=3, help='')

######################################################################################
#  File parameters
######################################################################################
parser.add_argument('--task', type=str, default='yelp', help='Specify datasets.')
parser.add_argument('--word_to_id_file', type=str, default='', help='')
parser.add_argument('--data_path', type=str, default='', help='')

######################################################################################
#  Model parameters
######################################################################################
parser.add_argument('--word_dict_max_num', type=int, default=5, help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--max_sequence_length', type=int, default=60)
parser.add_argument('--num_layers_AE', type=int, default=2)
parser.add_argument('--transformer_model_size', type=int, default=256)
parser.add_argument('--transformer_ff_size', type=int, default=1024)

parser.add_argument('--latent_size', type=int, default=256)
parser.add_argument('--word_dropout', type=float, default=1.0)
parser.add_argument('--embedding_dropout', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--label_size', type=int, default=1)


args = parser.parse_args()

# args.if_load_from_checkpoint = False
# args.if_load_from_checkpoint = True
# args.checkpoint_name = "1557667911"


######################################################################################
#  End of hyper parameters
######################################################################################


def add_log(ss):
    now_time = time.strftime("[%Y-%m-%d %H:%M:%S]: ", time.localtime())
    print(now_time + ss)
    with open(args.log_file, 'a') as f:
        f.write(now_time + str(ss) + '\n')
    return


def add_output(ss):
    with open(args.output_file, 'a') as f:
        f.write(str(ss) + '\n')
    return


def preparation():
    # set model save path
    if args.if_load_from_checkpoint:
        timestamp = args.checkpoint_name
    else:
        timestamp = str(int(time.time()))
        print("create new model save path: %s" % timestamp)
    args.current_save_path = 'save/%s/' % timestamp
    args.log_file = args.current_save_path + time.strftime("log_%Y_%m_%d_%H_%M_%S.txt", time.localtime())
    args.output_file = args.current_save_path + time.strftime("output_%Y_%m_%d_%H_%M_%S.txt", time.localtime())
    print("create log file at path: %s" % args.log_file)

    if os.path.exists(args.current_save_path):
        add_log("Load checkpoint model from Path: %s" % args.current_save_path)
    else:
        os.makedirs(args.current_save_path)
        add_log("Path: %s is created" % args.current_save_path)

    # set task type
    if args.task == 'yelp':
        args.data_path = '../../data/yelp/processed_files/'
    elif args.task == 'amazon':
        args.data_path = '../../data/amazon/processed_files/'
    elif args.task == 'imagecaption':
        pass
    else:
        raise TypeError('Wrong task type!')

    # prepare data
    args.id_to_word, args.vocab_size, \
    args.train_file_list, args.train_label_list = prepare_data(
        data_path=args.data_path, max_num=args.word_dict_max_num, task_type=args.task
    )
    return


def train_iters(ae_model, dis_model):
    train_data_loader = non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    train_data_loader.create_batches(args.train_file_list, args.train_label_list, if_shuffle=True)
    add_log("Start train process.")
    ae_model.train()
    dis_model.train()

    ae_optimizer = NoamOpt(ae_model.src_embed[0].d_model, 1, 2000,
                           torch.optim.Adam(ae_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=0.0001)

    ae_criterion = get_cuda(LabelSmoothing(size=args.vocab_size, padding_idx=args.id_pad, smoothing=0.1))
    dis_criterion = nn.BCELoss(size_average=True)

    for epoch in range(200):
        print('-' * 94)
        epoch_start_time = time.time()
        for it in range(train_data_loader.num_batch):
            batch_sentences, tensor_labels, \
            tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
            tensor_tgt_mask, tensor_ntokens = train_data_loader.next_batch()

            # Forward pass
            latent, out = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)

            # Loss calculation
            loss_rec = ae_criterion(out.contiguous().view(-1, out.size(-1)),
                                    tensor_tgt_y.contiguous().view(-1)) / tensor_ntokens.data

            ae_optimizer.optimizer.zero_grad()

            loss_rec.backward()
            ae_optimizer.step()

            # Classifier
            dis_lop = dis_model.forward(to_var(latent.clone()))

            loss_dis = dis_criterion(dis_lop, tensor_labels)

            dis_optimizer.zero_grad()
            loss_dis.backward()
            dis_optimizer.step()

            if it % 200 == 0:
                add_log(
                    '| epoch {:3d} | {:5d}/{:5d} batches | rec loss {:5.4f} | dis loss {:5.4f} |'.format(
                        epoch, it, train_data_loader.num_batch, loss_rec, loss_dis))

                print(id2text_sentence(tensor_tgt_y[0], args.id_to_word))
                generator_text = ae_model.greedy_decode(latent,
                                                        max_len=args.max_sequence_length,
                                                        start_id=args.id_bos)
                print(id2text_sentence(generator_text[0], args.id_to_word))

        add_log(
            '| end of epoch {:3d} | time: {:5.2f}s |'.format(
                epoch, (time.time() - epoch_start_time)))
        # Save model
        torch.save(ae_model.state_dict(), args.current_save_path + 'ae_model_params.pkl')
        torch.save(dis_model.state_dict(), args.current_save_path + 'dis_model_params.pkl')
    return


def eval_iters(ae_model, dis_model):
    eval_data_loader = non_pair_data_loader(
        batch_size=1, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size
    )
    eval_file_list = [
        args.data_path + 'sentiment.test.0',
        args.data_path + 'sentiment.test.1',
    ]
    eval_label_list = [
        [0],
        [1],
    ]
    eval_data_loader.create_batches(eval_file_list, eval_label_list, if_shuffle=False)
    gold_ans = load_human_answer(args.data_path)
    assert len(gold_ans) == eval_data_loader.num_batch


    add_log("Start eval process.")
    ae_model.eval()
    dis_model.eval()
    for it in range(eval_data_loader.num_batch):
        batch_sentences, tensor_labels, \
        tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
        tensor_tgt_mask, tensor_ntokens = eval_data_loader.next_batch()

        print("------------%d------------" % it)
        print(id2text_sentence(tensor_tgt_y[0], args.id_to_word))
        print("origin_labels", tensor_labels)

        latent, out = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)
        generator_text = ae_model.greedy_decode(latent,
                                                max_len=args.max_sequence_length,
                                                start_id=args.id_bos)
        print(id2text_sentence(generator_text[0], args.id_to_word))

        # Define target label
        target = get_cuda(torch.tensor([[1.0]], dtype=torch.float))
        if tensor_labels[0].item() > 0.5:
            target = get_cuda(torch.tensor([[0.0]], dtype=torch.float))
        print("target_labels", target)

        modify_text = fgim_attack(dis_model, latent, target, ae_model, args.max_sequence_length, args.id_bos,
                                        id2text_sentence, args.id_to_word, gold_ans[it])
        add_output(modify_text)
    return



if __name__ == '__main__':
    preparation()

    ae_model = get_cuda(make_model(d_vocab=args.vocab_size,
                                   N=args.num_layers_AE,
                                   d_model=args.transformer_model_size,
                                   latent_size=args.latent_size,
                                   d_ff=args.transformer_ff_size,
    ))
    dis_model = get_cuda(Classifier(latent_size=args.latent_size, output_size=args.label_size))

    if args.if_load_from_checkpoint:
        # Load models' params from checkpoint
        ae_model.load_state_dict(torch.load(args.current_save_path + 'ae_model_params.pkl'))
        dis_model.load_state_dict(torch.load(args.current_save_path + 'dis_model_params.pkl'))
    else:
        train_iters(ae_model, dis_model)

    eval_iters(ae_model, dis_model)

    print("Done!")

