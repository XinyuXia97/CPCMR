import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm, Dropout, MultiheadAttention, ModuleList, BatchNorm1d
from torch.nn.functional import normalize
from torch.nn import functional as F

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()+ eps
    X = torch.div(X, norm)
    return X


class FuseTransEncoder(nn.Module):
    def __init__(self,  num_layers, hidden_size, nhead, num_class):
        super(FuseTransEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        self.transformerEncoder = TransformerEncoder(encoder_layer= encoder_layer, num_layers= num_layers)
        self.Linear1 = nn.Linear(1024, num_class)
        self.Linear2 = nn.Linear(1024, num_class)

        self.dropout = Dropout(p=0.1)
        self.linear1 = nn.Linear(hidden_size, 4096)
        self.linear2 = nn.Linear(4096, hidden_size)
        self.dropout3 = Dropout(p=0.1)
        self.dropout4 = Dropout(p=0.1)
        self.norm = LayerNorm(hidden_size , eps=1e-5)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout3(x)
    def forward(self, tokens):
        # encoder_X = self._ff_block(tokens)
        encoder_X = self.transformerEncoder(tokens)
        encoder_X_r = encoder_X.reshape( -1,2048)
        # encoder_X_r = normalize(encoder_X_r, p =2 ,dim =0) 
        encoder_X_r = normalize(encoder_X_r, p =2 ,dim =1) 
       
        cls_out_image = self.Linear1(encoder_X_r[:,:1024]) 
        cls_out_text = self.Linear2(encoder_X_r[:,1024:2048]) 

        encoder_X = encoder_X_r.reshape_as(encoder_X)
        img_recon = encoder_X[:, :, :1024]
        text_recon = encoder_X[:, :, 1024:]
        return img_recon, text_recon , cls_out_image, cls_out_text


class Prototype(nn.Module):
    '''
        可学习原型
    '''
    def __init__(self, feat_dim, n_classes, device):
        super(Prototype, self).__init__()
        self.anchor = nn.Parameter(torch.randn(n_classes, feat_dim), requires_grad=True).to(device)

        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.anchor, mode='fan_out')
       
    def forward(self):
        self.anchor.data = l2norm(self.anchor.data, dim=0)
        return self.anchor


class SelfAttentionLayer(nn.Module):
    def __init__(self,  num_layers, hidden_size, nhead):
        super(SelfAttentionLayer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        self.selfAttention = TransformerEncoder(encoder_layer= encoder_layer, num_layers= num_layers)
    def forward(self, tokens):
        encoder_X = self.selfAttention(tokens)
        return encoder_X
        

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attention = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, 2048)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(2048, d_model)
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.norm3 = LayerNorm(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, anchor):
        src = self.norm3(src)
        src2 = self.cross_attention(src, anchor, anchor)[0] 
        # src2 = self.cross_attention(src, src, src)[0] 
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
     
        return src
   

class CrossAttentionEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nhead):
        super(CrossAttentionEncoder, self).__init__()
        self.self_att = SelfAttentionLayer(num_layers=num_layers, hidden_size= hidden_size, nhead = nhead)
        self.cross_att = CrossAttentionLayer(d_model= hidden_size, nhead= nhead)

        self.dropout = Dropout(p=0.1)
        self.linear1 = nn.Linear(hidden_size, 2048)
        self.linear2 = nn.Linear(2048, hidden_size)
        self.dropout3 = Dropout(p=0.1)
        self.dropout4 = Dropout(p=0.1)
        self.norm = LayerNorm(hidden_size , eps=1e-5)
        
    def _ff_block(self, x):
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout3(x)
    def forward(self, x , anchor):
        output = x
        # output = self.self_att(output)
        output = self._ff_block(output)  
        output = self.cross_att(output,anchor)

        
        return output



class ImageTransEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nhead, num_class):
        super(ImageTransEncoder, self).__init__()
        self.transformerEncoder = CrossAttentionEncoder(num_layers, hidden_size, nhead)
       
        
    def forward(self, X, anchor):  
        anchor = torch.unsqueeze(anchor, 1)
        anchor  = anchor.repeat(1, X.shape[1], 1) 
        output = self.transformerEncoder(X, anchor)
        output_r = output.reshape(-1,1024)
        output_r = normalize(output_r, p=2,  dim=1)
        output = output_r.reshape_as(output)
        return output


class TextTransEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nhead,num_class):
        super(TextTransEncoder, self).__init__()
        self.transformerEncoder = CrossAttentionEncoder(num_layers, hidden_size, nhead)
       
    def forward(self, X, anchor):  
        anchor = torch.unsqueeze(anchor, 1)
        anchor  = anchor.repeat(1, X.shape[1], 1) 
        output = self.transformerEncoder(X, anchor)
        output_r = output.reshape(-1,1024)
        output_r = normalize(output_r, p=2,  dim=1)
        output = output_r.reshape_as(output)
        return output
