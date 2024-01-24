import torch
import torch.nn as nn
import pickle

# load the data file
Data = pickle.load(open('./model/Data.pkl', 'rb'))

# load word2index from the Data
word2index = Data['word2index']

# Skipgram Model
class Skipgram (nn.Module):
    
    def __init__(self, voc_size, emb_size):
        super(Skipgram, self).__init__()
        self.embedding_center  = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
    
    def forward(self, center_word, outside_word, all_vocabs):
        
        # embedding for all: result as a vector for each 
        center_embed  = self.embedding_center(center_word)   # (batch_size, 1, emb_size)
        outside_embed = self.embedding_outside(outside_word) # (batch_size, 1, emb_size)
        all_embed     = self.embedding_outside(all_vocabs)   # (batch_size, voc_size, emb_size)
        
        # write the equation
        top_term = torch.exp(outside_embed.bmm(center_embed.transpose(1,2)).squeeze(2))
        # (batch_size, 1, emb_size) @ (batch_size, emb_size, 1) = (batch_size, 1, 1) = (bacth_size, 1)
        
        lower_term = all_embed.bmm(center_embed.transpose(1,2)).squeeze(2)
        # (batch_size, voc_size, emb_size) @ (batch_size, emb_size, 1) = (batch_size, voc_size, 1) = (batch_size, voc_size)
        lower_term_sum = torch.sum(torch.exp(lower_term), 1)
        # (batch_size, 1)
        
        loss = - torch.mean(torch.log(top_term / lower_term_sum))
        
        return loss
    
    # function to get the embedding given a word
    def get_embed(self, word):
        id_tensor = torch.LongTensor([word2index[word]])
        c_embed = self.embedding_center(id_tensor)
        o_embed = self.embedding_outside(id_tensor)
        word_embed = (c_embed + o_embed) / 2
        # x,y = word_embed[0][0].item(), word_embed[0][1].item()
        return word_embed
    
# Skipgram (Negative Sampling) Model

class SkipgramNeg (nn.Module):
    
    def __init__(self, voc_size, emb_size):
        super(SkipgramNeg, self).__init__()
        self.embedding_center  = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
        self.logsigmoid        = nn.LogSigmoid()
    
    def forward(self, center, outside, negative_words):
        
        # center, outside : (bs, 1)
        # negative : (bs, k)
        
        center_embed  = self.embedding_center(center) # (bs, 1, emb_size)
        outside_embed = self.embedding_outside(outside) #(bs, 1, emb_size)
        neg_embed     = self.embedding_outside(negative_words) # (bs, k, emb_size)
        
        uovc          = outside_embed.bmm(center_embed.transpose(1,2)).squeeze(2) # (bs,1)
        ukvc          = -neg_embed.bmm(center_embed.transpose(1,2)).squeeze(2) #(bs,k)
        ukvc_sum      = torch.sum(ukvc, 1).reshape(-1,1) # sum across k , reshape>> (batch_size,1)
        
        loss = self.logsigmoid(uovc) + self.logsigmoid(ukvc_sum)
        
        return -torch.mean(loss)        
        

# GloVe Model

class Glove(nn.Module):
    
    def __init__(self, voc_size, emb_size):
        super(Glove, self).__init__()
        self.center_embedding  = nn.Embedding(voc_size, emb_size)
        self.outside_embedding = nn.Embedding(voc_size, emb_size)
        
        self.center_bias       = nn.Embedding(voc_size, 1)
        self.outside_bias      = nn.Embedding(voc_size, 1) 
        
    def forward(self, center, outside, cooc, weighting):
        center_embeds  = self.center_embedding(center)   # (batch_size, 1, embed_size)
        outside_embeds = self.outside_embedding(outside) # (batch_size, 1, embed_size)
        
        center_bias    = self.center_bias(center).squeeze(1) # (batch_size, voc_size)
        target_bias    = self.outside_bias(outside).squeeze(1)
        
        inner_product = outside_embeds.bmm(center_embeds.transpose(1,2)).squeeze(2)
        # (batch_size, 1, emb_size) @ (batch_size, emb_size, 1) = (batch_size, 1, 1) = (batch_size, 1)
        
        loss = weighting * torch.pow(inner_product + center_bias + target_bias - cooc, 2)
        
        return torch.sum(loss)
