import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from tqdm import tqdm
import math

'''
mask
(1) padding-mask
- mask padding
- multiplied with attention matrix

(2) look-ahead mask
- multiplied with Q@K_T matrix
'''
def makeMask(tensor, option:str) -> torch.Tensor:
    '''
    tensor (bs, item_len)
    '''
    if option == 'padding':
        tmp = torch.full_like(tensor, fill_value = 0)
        # (bs, item_len)

        mask = (tensor != tmp).float() # real items
        # (bs, item_len)
        # 1 for real items
        # 0 for paddings

        mask = rearrange(mask, 'bs item_len -> bs 1 1 item_len')
        # (bs, 1, 1, item_len)
        # (1, 1) is for broadcasting 
    
    elif option == 'lookahead':
        # tensor: (bs, item_len)

        padding_mask = makeMask(tensor, 'padding') 
        # (bs, 1, 1, item_len)
        padding_mask = repeat(padding_mask, 'bs 1 1 item_len -> bs 1 new item_len', new= padding_mask.shape[3])
        # (bs, 1, item_len, item_len)

        mask = torch.ones_like(padding_mask)
        mask = torch.tril(mask)
        # Returns the lower triangular part of the matrix (2-D tensor) 
        # or batch of matrices input, the other elements of the result tensor out are set to 0.
        mask = mask*padding_mask    

    return mask

'''
Multi-head self attention
'''

class MSA(nn.Module):

    def __init__(self, hidden_dim: int = 512, num_head: int = 8):
        super().__init__()

        # embedding_dim
        self.hidden_dim = hidden_dim
        self.num_head = num_head

        self.head_dim = hidden_dim // num_head
        self.scale = torch.sqrt(torch.FloatTensor())

        self.genQ = nn.Linear(hidden_dim, hidden_dim)
        self.genK = nn.Linear(hidden_dim, hidden_dim)
        self.genV = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, srcQ, srcK, srcV, mask= None):
        Q, K, V = self.genQ(srcQ), self.genK(srcK), self.genV(srcV)
        # hidden_dim = num_head * head_dim
        Q = rearrange(Q, 'bs item_len (num_head head_dim) -> bs num_head item_len head_dim', 
                        num_head= self.num_head)
        K_T = rearrange(K, 'bs item_len (num_head head_dim) -> bs num_head head_dim item_len', 
                        num_head= self.num_head)
        V = rearrange(V, 'bs item_len (num_head head_dim) -> bs num_head item_len head_dim', 
                        num_head= self.num_head)

        A = Q@K_T/math.sqrt(self.head_dim)
        if mask is not None:
            '''
            mask.shape 
            if padding : (bs, 1, 1, item_len)
            elif lookahead: (bs, 1, item_len, item_len)
            '''
            A = torch.masked_fill(A, (mask==0), -1e+4)
            # fill -e+4 for attention values of paddings
        
        A = torch.softmax(A, dim = -1)
        result = self.dropout(A)@V # (bs num_head, item_len, head_dim)
        result = rearrange(result, 'bs num_head item_len head_dim ->  bs item_len (num_head head_dim)')
        # projection
        result = self.fc_out(result)
        
        return result

'''
Pointwise feed-forward network
'''
class FFN(nn.Module):

    def __init__(self, hidden_dim: int = 512, inner_dim:int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim

        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input):
        out = self.act(self.fc1(input))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class EncoderLayer(nn.Module):

    def __init__(self, hidden_dim, num_head, inner_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.msa = MSA(hidden_dim=hidden_dim, num_head=num_head)
        self.ffn = FFN(inner_dim = inner_dim, hidden_dim = hidden_dim)
        self.layerNorm1 = nn.LayerNorm(hidden_dim)
        self.layerNorm2 = nn.LayerNorm(hidden_dim)
        
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, input, mask= None):
        # multi-head attetion
        output = self.msa(input, input, input, mask= mask)
        # dropout
        output = self.drop1(output)
        # add&norm
        output = input + output
        output = self.layerNorm1(output)
        # ffc
        output_2 = self.ffn(output)
        output_2 = self.drop2(output_2)
        # add&norm 
        output = output + output_2
        output = self.layerNorm2(output)
        # (bs, item_len, hidden_dim)
        return output
