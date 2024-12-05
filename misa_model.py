# Code adapted from https://github.com/declare-lab/MISA/blob/master/src/models.py
import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import math
from transformers import BertTokenizer, BertModel, BertConfig

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)



# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self, mosi_or_mosei, output_dim=0, dropout=0.5, activation='relu', rnncell='lstm', hidden_size=128, use_cmd_sim=True, reverse_grad_weight=1.0, use_bert=False, encode_batch=None):
        super(MISA, self).__init__()
        self.use_cmd_sim = use_cmd_sim
        self.reverse_grad_weight = reverse_grad_weight
        self.use_bert = use_bert
        if encode_batch is None:
            if self.use_bert:
                self.encode_batch = True
            else:
                self.encode_batch = False
        elif type(encode_batch) != bool:
            raise ValueError('encode_batch should be a boolean')
        if mosi_or_mosei == 'mosi':
            self.text_size = 300
            self.visual_size = 47
            self.acoustic_size = 74
        elif mosi_or_mosei == 'mosei':
            self.text_size = 300
            self.visual_size = 713
            self.acoustic_size = 74
        else:
            raise ValueError(f'Unknown dataset: {mosi_or_mosei}')

        self.hidden_size = output_dim

        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = output_dim
        self.output_dim = output_dim *3
        self.dropout_rate = dropout_rate = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        
        self.rnncell = rnncell
        rnn = nn.LSTM if rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between

        if self.use_bert:
            from transformers import BertTokenizer
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # Initializing a BERT bert-base-uncased style configuration
            bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        else:
            # self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
            self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True, batch_first=True)
            self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True, batch_first=True)
        
        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True, batch_first=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True, batch_first=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True, batch_first=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True, batch_first=True)



        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=self.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(self.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=self.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(self.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1]*4, out_features=self.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(self.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2]*4, out_features=self.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(self.hidden_size))


        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())


        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))



        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=self.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=self.hidden_size, out_features=4))



        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.hidden_size*6, out_features=self.hidden_size*3))
        # self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        # self.fusion.add_module('fusion_layer_1_activation', self.activation)
        # self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.hidden_size*3, out_features= output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))


        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.critic = nn.Sequential(
            nn.Linear(self.output_dim*2, 8),
        )

        

        
    def extract_features(self, sequence, rnn1, rnn2, layer_norm, lengths=None):
        if lengths is not None:
            lengths = lengths.to('cpu')
        if lengths is not None:
            packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        else:
            packed_sequence = sequence

        if self.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        if lengths is not None:
            padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True)
        else:
            padded_h1 = packed_h1
        normed_h1 = layer_norm(padded_h1)
        if lengths is not None:
            packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)
        else:
            packed_normed_h1 = normed_h1

        if self.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, batch):

        if self.use_bert:
            visual, acoustic, sentences = batch 
            # Find the max length of the sentence
            bert_details = []
            for sample in sentences:
                text = " ".join(sample)
                encoded_bert_sent = self.bert_tokenizer.encode_plus(
                    text, max_length=512, add_special_tokens=True, padding='max_length')
                bert_details.append(encoded_bert_sent)  
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details]).to(device)
            bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details]).to(device)
            bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details]).to(device)          
            # visual, acoustic, sentences, bert_sentences, bert_sentence_types, bert_sentence_att_mask, lengths = batch 
            batch_size = visual.shape[0]
            bert_output = self.bertmodel(input_ids=bert_sentences, 
                                         attention_mask=bert_sentence_att_mask, 
                                         token_type_ids=bert_sentence_types)      
            bert_output = bert_output[0]

            # masked mean
            masked_output = torch.mul(bert_sentence_att_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sentence_att_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

            lengths = None

            utterance_text = bert_output
        else:
            visual, acoustic, sentences = batch
            lengths = None
            batch_size = visual.shape[0]
            # extract features from text modality
            sentences = self.embed(sentences)
            final_h1t, final_h2t = self.extract_features(sentences, self.trnn1, self.trnn2, self.tlayer_norm)
            utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        
        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, self.vrnn1, self.vrnn2, self.vlayer_norm, lengths)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, self.arnn1, self.arnn2, self.alayer_norm, lengths)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)


        if not self.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None


        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        # self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        o = self.fusion(h)
        return o
    
    # def reconstruct(self,):

    #     self.utt_t = (self.utt_private_t + self.utt_shared_t)
    #     self.utt_v = (self.utt_private_v + self.utt_shared_v)
    #     self.utt_a = (self.utt_private_a + self.utt_shared_a)

    #     self.utt_t_recon = self.recon_t(self.utt_t)
    #     self.utt_v_recon = self.recon_v(self.utt_v)
    #     self.utt_a_recon = self.recon_a(self.utt_a)


    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)


        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)


    def forward(self, batch):
        o = self.alignment(batch)
        return o

    def score(self, u, v):
        concat = torch.cat((u, v), dim=1)
        return self.critic(concat)
    
    def encode_modalities(self, batch):

        if self.use_bert:
            visual, acoustic, sentences = batch 
            # Find the max length of the sentence
            bert_details = []
            for sample in sentences:
                text = " ".join(sample)
                encoded_bert_sent = self.bert_tokenizer.encode_plus(
                    text, max_length=512, add_special_tokens=True, padding='max_length')
                bert_details.append(encoded_bert_sent)  
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details]).to(device)
            bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details]).to(device)
            bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details]).to(device)          
            # visual, acoustic, sentences, bert_sentences, bert_sentence_types, bert_sentence_att_mask, lengths = batch 
            batch_size = visual.shape[0]
            bert_output = self.bertmodel(input_ids=bert_sentences, 
                                         attention_mask=bert_sentence_att_mask, 
                                         token_type_ids=bert_sentence_types)      
            bert_output = bert_output[0]

            # masked mean
            masked_output = torch.mul(bert_sentence_att_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sentence_att_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

            lengths = None

            utterance_text = bert_output
        else:
            visual, acoustic, sentences = batch
            lengths = None
            batch_size = visual.shape[0]
            # extract features from text modality
            sentences = self.embed(sentences)
            final_h1t, final_h2t = self.extract_features(sentences, self.trnn1, self.trnn2, self.tlayer_norm)
            utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        
        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, self.vrnn1, self.vrnn2, self.vlayer_norm, lengths)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, self.arnn1, self.arnn2, self.alayer_norm, lengths)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        utterances = [utterance_video, utterance_audio, utterance_text]
        return utterances

    def fuse(self, batch):
        utterance_video, utterance_audio, utterance_text = batch
        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)


        if not self.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None


        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        # self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        o = self.fusion(h)
        return o
    
if __name__ == '__main__':
    model = MISA('mosi', output_dim=128, use_bert=True)
    print(model)