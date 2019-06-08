import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import math, copy, time
import torch.nn.utils.rnn as rnn_utils
from data import get_cuda, to_var, calc_bleu
import numpy as np


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention' """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


################ Encoder ################
class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


################ Decoder ################
class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


################ Generator ################
class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class AttentionScore(nn.Module):
    """
    correlation_func = 1, sij = x1^Tx2
    correlation_func = 2, sij = (Wx1)D(Wx2)
    correlation_func = 3, sij = Relu(Wx1)DRelu(Wx2)
    correlation_func = 4, sij = x1^TWx2
    correlation_func = 5, sij = Relu(Wx1)DRelu(Wx2)
    """

    def __init__(self, input_size, hidden_size, correlation_func=1, do_similarity=False):
        super(AttentionScore, self).__init__()
        self.correlation_func = correlation_func
        self.hidden_size = hidden_size

        if correlation_func == 2 or correlation_func == 3:
            self.linear = nn.Linear(input_size, hidden_size, bias=False)
            if do_similarity:
                self.diagonal = Parameter(torch.ones(1, 1, 1) / (hidden_size ** 0.5), requires_grad=False)
            else:
                self.diagonal = Parameter(torch.ones(1, 1, hidden_size), requires_grad=True)

        if correlation_func == 4:
            self.linear = nn.Linear(input_size, input_size, bias=False)

        if correlation_func == 5:
            self.linear = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x1, x2, x2_mask):
        '''
        Input:
        x1: batch x word_num1 x dim
        x2: batch x word_num2 x dim
        Output:
        scores: batch x word_num1 x word_num2
        '''
        # x1 = dropout(x1, p = dropout_p, training = self.training)
        # x2 = dropout(x2, p = dropout_p, training = self.training)

        x1_rep = x1
        x2_rep = x2
        batch = x1_rep.size(0)
        word_num1 = x1_rep.size(1)
        word_num2 = x2_rep.size(1)
        dim = x1_rep.size(2)
        if self.correlation_func == 2 or self.correlation_func == 3:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)  # Wx1
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)  # Wx2
            if self.correlation_func == 3:
                x1_rep = F.relu(x1_rep)
                x2_rep = F.relu(x2_rep)
            x1_rep = x1_rep * self.diagonal.expand_as(x1_rep)
            # x1_rep is (Wx1)D or Relu(Wx1)D
            # x1_rep: batch x word_num1 x dim (corr=1) or hidden_size (corr=2,3)

        if self.correlation_func == 4:
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, dim)  # Wx2

        if self.correlation_func == 5:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)  # Wx1
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)  # Wx2
            x1_rep = F.relu(x1_rep)
            x2_rep = F.relu(x2_rep)
        scores = x1_rep.bmm(x2_rep.transpose(1, 2))
        empty_mask = x2_mask.eq(0).expand_as(scores)
        scores.data.masked_fill_(empty_mask.data, -float('inf'))
        # softmax
        alpha_flat = F.softmax(scores, dim=-1)
        return alpha_flat


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, gru, src_embed, tgt_embed, generator, input_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.gru = gru
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.linear = nn.Linear(input_size * 2, input_size)
        self.attention = AttentionScore(input_size, input_size)
        self.gru_decoder = nn.GRU(input_size * 2, input_size, 1)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        """
        memory = self.encode(src, src_mask)  # (batch_size, max_src_seq, d_model)
        # attented_mem=self.attention(memory,memory,memory,src_mask)
        # memory=attented_mem
        score = self.attention(memory, memory, src_mask)
        attent_memory = score.bmm(memory)
        # memory=self.linear(torch.cat([memory,attent_memory],dim=-1))

        memory, _ = self.gru(attented_mem)
        '''
        score=torch.sigmoid(self.linear(memory))
        memory=memory*score
        '''
        latent = torch.sum(memory, dim=1)  # (batch_size, d_model)
        logit = self.decode(latent.unsqueeze(1), tgt, tgt_mask)  # (batch_size, max_tgt_seq, d_model)
        # logit,_=self.gru_decoder(logit)
        prob = self.generator(logit)  # (batch_size, max_seq, vocab_size)
        return latent, prob

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, tgt, tgt_mask):
        # memory: (batch_size, 1, d_model)
        src_mask = get_cuda(torch.ones(memory.size(0), 1, 1).long())
        # print("src_mask here", src_mask)
        # print("src_mask", src_mask.size())
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def greedy_decode(self, latent, max_len, start_id):
        '''
        latent: (batch_size, max_src_seq, d_model)
        src_mask: (batch_size, 1, max_src_len)
        '''
        batch_size = latent.size(0)
        ys = get_cuda(torch.ones(batch_size, 1).fill_(start_id).long())  # (batch_size, 1)
        for i in range(max_len - 1):
            # input("==========")
            # print("="*10, i)
            # print("ys", ys.size())  # (batch_size, i)
            # print("tgt_mask", subsequent_mask(ys.size(1)).size())  # (1, i, i)
            out = self.decode(latent.unsqueeze(1), to_var(ys), to_var(subsequent_mask(ys.size(1)).long()))
            prob = self.generator(out[:, -1])
            # print("prob", prob.size())  # (batch_size, vocab_size)
            _, next_word = torch.max(prob, dim=1)
            # print("next_word", next_word.size())  # (batch_size)

            # print("next_word.unsqueeze(1)", next_word.unsqueeze(1).size())

            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
            # print("ys", ys.size())
        return ys[:, 1:]


def make_model(d_vocab, N, d_model, d_ff=1024, h=4, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.GRU(d_model, d_model, 1),
        nn.Sequential(Embeddings(d_model, d_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, d_vocab), c(position)),
        Generator(d_model, d_vocab),
        d_model
    )
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Classifier(nn.Module):
    def __init__(self, latent_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, 100)
        self.relu1 = nn.LeakyReLU(0.2, )
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(50, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.fc1(input)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)

        # out = F.log_softmax(out, dim=1)
        return out  # batch_size * label_size


def fgim_attack(model, origin_data, target, ae_model, max_sequence_length, id_bos,
                id2text_sentence, id_to_word, gold_ans):
    """Fast Gradient Iterative Methods"""

    dis_criterion = nn.BCELoss(size_average=True)

    gold_text = id2text_sentence(gold_ans, id_to_word)
    print("gold:", gold_text)
    # while True:
    for epsilon in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        it = 0
        data = origin_data
        while True:
            print("epsilon:", epsilon)

            data = to_var(data.clone())  # (batch_size, seq_length, latent_size)
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True
            output = model.forward(data)
            # Calculate gradients of model in backward pass
            # print("target", target[0].item())
            # print("output", output[0].item())
            loss = dis_criterion(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            # print("data_grad")
            # print(data_grad)
            data = data - epsilon * data_grad
            # print("epsilon * data_grad")
            # print((epsilon * data_grad))
            # print("data")
            # print(data)
            # print("perturbed_data")
            # print(perturbed_data)
            it += 1
            # data = perturbed_data
            epsilon = epsilon * 0.9

            generator_id = ae_model.greedy_decode(data,
                                                    max_len=max_sequence_length,
                                                    start_id=id_bos)
            generator_text = id2text_sentence(generator_id[0], id_to_word)
            print("| It {:2d} | dis model pred {:5.4f} |".format(it, output[0].item()))
            print(generator_text)
            if it >= 5:
                break
    return



if __name__ == '__main__':
    # plt.figure(figsize=(15, 5))
    # pe = PositionalEncoding(20, 0)
    # y = pe.forward(Variable(torch.zeros(1, 100, 20)))
    # plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    # plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    # plt.show()

    # Small example model.
    # tmp_model = make_model(10, 10, 2)
    pass
