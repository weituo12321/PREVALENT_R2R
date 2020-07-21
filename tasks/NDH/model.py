
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from r2rmodel import BertImgEncoder


class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx,decoder_init,c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class R2RSoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, ctx_hidden_size, dim):
        '''Initialize layer.'''
        super(R2RSoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, ctx_hidden_size, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim + ctx_hidden_size, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None, ctx_drop=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim

        if ctx_drop is not None:
            weighted_context = ctx_drop(weighted_context)

        h_tilde = torch.cat((weighted_context, h), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn

class VisualSoftDotAttention(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, h_dim, v_dim, dot_dim=256):
        '''Initialize layer.'''
        super(VisualSoftDotAttention, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, visual_context, mask=None):
        '''Propagate h through the network.

        h: batch x h_dim
        visual_context: batch x v_num x v_dim 100x36x2048
        '''
        target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1
        context = self.linear_in_v(visual_context)  # batch x v_num x dot_dim

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x v_num
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num

        weighted_context = torch.bmm(attn3, visual_context).squeeze(1)  # batch x v_dim
        return weighted_context, attn


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size,
                      dropout_ratio, feature_size=2048):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, output_action_size)

    def forward(self, action, feature, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        action_embeds = self.embedding(action)   # (batch, 1, embedding_size)
        action_embeds = action_embeds.squeeze()
        concat_input = torch.cat((action_embeds, feature), 1) # (batch, embedding_size+feature_size)
        drop = self.drop(concat_input)
        h_1,c_1 = self.lstm(drop, (h_0,c_0))
        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
        logit = self.decoder2action(h_tilde)
        return h_1,c_1,alpha,logit

class EltwiseProdScoring(nn.Module):
    '''
    Linearly mapping h and v to the same dimension, and do an elementwise
    multiplication and a linear scoring
    '''

    def __init__(self, h_dim, a_dim, dot_dim=256):
        '''Initialize layer.'''
        super(EltwiseProdScoring, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_a = nn.Linear(a_dim, dot_dim, bias=True)
        self.linear_out = nn.Linear(dot_dim, 1, bias=True)

    def forward(self, h, all_u_t, mask=None):
        '''Propagate h through the network.

        h: batch x h_dim
        all_u_t: batch x a_num x a_dim
        '''
        target = self.linear_in_h(h).unsqueeze(1)  # batch x 1 x dot_dim
        context = self.linear_in_a(all_u_t)  # batch x a_num x dot_dim
        eltprod = torch.mul(target, context)  # batch x a_num x dot_dim
        logits = self.linear_out(eltprod).squeeze(2)  # batch x a_num
        return logits

class R2RAttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, input_action_size, output_action_size, embedding_size, ctx_hidden_size, hidden_size,
                      dropout_ratio, feature_size, panoramic, action_space, dec_h_type):
        super(R2RAttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size + (128 if panoramic else 0)
        self.ctx_hidden_size = ctx_hidden_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.panoramic = panoramic
        self.action_space = action_space
        self.att_ctx_merge = None

        self.dec_h_type = dec_h_type

        action_hidden_size = hidden_size

        if not self.panoramic:
            LSTM_n_in = embedding_size + feature_size  # action_embedding + original single view feature
        else:
            if self.action_space == 6:
                LSTM_n_in = embedding_size + feature_size + self.feature_size  # attented multi-view feature (single feature +128)
            else:  # if self.action_space == -1:
                LSTM_n_in = self.feature_size * 2  # action feature + attented multi-view feature
                #LSTM_n_in = feature_size + self.feature_size * 2  # img feature + action feature + attented multi-view feature

        self.lstm = nn.LSTMCell(LSTM_n_in, hidden_size)
        self.attention_layer = R2RSoftDotAttention(ctx_hidden_size, hidden_size)
        # panoramic feature
        if self.panoramic:
            self.visual_attention_layer = VisualSoftDotAttention(hidden_size, self.feature_size)
        else:
            self.visual_attention_layer = None

        # panoramic action space
        self.u_begin, self.embedding = None, None
        if self.action_space == 6:
            self.embedding = nn.Embedding(input_action_size, embedding_size)
            self.decoder2action = nn.Linear(action_hidden_size, output_action_size)
        else:
            self.u_begin = Variable(torch.zeros(self.feature_size), requires_grad=False).cuda()
            self.decoder2action = EltwiseProdScoring(action_hidden_size, self.feature_size)
            #self.decoder2action = EltwiseProdScoring(action_hidden_size+self.feature_size, self.feature_size) # debug

        self.decoder2feature = None

    def forward(self, action_prev, u_prev, u_features,  # teacher_u_feature,
                feature, feature_all, h_0, c_0, ctx, ctx_mask=None):  #, action_prev_feature
        ''' Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''

        if self.panoramic:  # feature.dim()==3:
            feature2, alpha_v = self.visual_attention_layer(h_0, feature_all)

        if self.action_space == 6:
            if self.panoramic:
                feature = torch.cat((feature, feature2), 1)
            action_embeds = self.embedding(action_prev.view(-1, 1))  # (batch, 1, embedding_size)
            action_embeds = action_embeds.squeeze()
        else: # bug: todo
            #feature = feature2
            action_embeds = u_prev

        concat_input = torch.cat((action_embeds, feature2), 1)
        #concat_input = torch.cat((feature, action_embeds, feature2), 1)
        drop = self.drop(concat_input)

        h_1, c_1 = self.lstm(drop, (h_0, c_0))
        h_1_drop = self.drop(h_1)


        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        if self.action_space == 6:
            logit = self.decoder2action(h_tilde)  # (100, 6)
        else:
            logit = self.decoder2action(h_tilde, u_features)
            #logit = self.decoder2action(action_input, u_features)

        pred_f = None

        if self.dec_h_type == 'vc':
            return h_tilde, c_1, alpha, logit, pred_f  # h_tilde
        else:
            return h_1, c_1, alpha, logit, pred_f # old verion



