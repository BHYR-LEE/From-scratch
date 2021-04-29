import torch
import torch.nn as nn
import os

print(os.path.relpath(os.path.abspath('.')))

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):  
        super(SelfAttention,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size ), "Use other number"

        self.values = nn.Linear(self.head_dim,self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim,self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim,self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.head_dim*heads, embed_size, bias=False)
    
    def forward(self, vlaues, keys, queries, mask):
        ## v,k,q = (N,seq_len,embed_dim)
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        ##split embedding by heads size   -> (N,seq_len, heads, head_dim)
        values = values.reshape(N, seq_len, self.heads, self.head_dim)
        keys = keys.reshape(N, seq_len, self.heads, self.head_dim)
        queries = queries.reshape(N, seq_len, self.heads, self.head_dim)

        #(16,10,8,32)   --> 8개의 10곱하기 10

        energy = torch.einsum("nqhd,nkhd->nhqk",[queries, keys])
        ## q : (N,q_len,heads,head_dim)
        ## k : (N,k_len,heads,head_dim)
        ## energy : (N,heads, q_len, k_len)  ## how much attention 필요하다

        if mask is not None:
            energy = energy.masked_fill_(maks == 0, float("-1e20") )

        attention = torch.softmax(energy/ self.embed_size ** (1/2) ,dim=3)  ## target sentence에 대해서 softmax


        out = torch.einsum("nhql,nlhd->nlhd",[attention,values]).reshape(N, query_len, -1)   ## concat
        ## attention : (N, heads, q_len, k_len)
        ## values : (N,val_len,heads, head_dim) 
        ## output : (N,query_len, heads, head_dim) -> 이후에 heads dim에서 concat

        out = self.fc_out(out)
        return out


    '''
    def forward(self,x):
        attentions = []
        heads = torch.split(x,self.head_dim,dim = -1 )

        for head in heads:
            values = self.values[head]
            keys = self.keys[head]
            queries = self.queries[head]

            one_attention = nn.Softmax(values*torch.matmul(queries,keys.T)/torch.sqrt(self.head_dim))

            

            attentions += [one_attention]

        torch.cat(attentions,dim=-1)

        return attention
    '''

class TransformerBlock(nn.Module):
    def __init__(self,embed_size, heads, dropout, forward_expansion):
        self.attention = SelfAttention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size,embed_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size*forward_expansion,embed_size)
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):

        ## 여기서 value key query 만드는 linear layer 필요할듯

        attention = self.attention(value,key,query,mask)
        x = self.dropout(self.norm1(attention + query))   ## 왜 쿼리를 더하지?
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N,seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))  ## 워드 임베딩 + 포지셔널 임베딩

        for layer in self.layers:
            out = layer(out, out, out, mask)   

        return out

class DecoderBlock(nn.Moduel):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(Decoder,self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x, trg_mask)
        query = self.dropout(self.nrom(attention + x))
        out = self.transformer(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, haeds, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size,heads,forward_expansion,dropout,device) for _ in range(num_layers)]
        )
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_lenth = x.shape
        positions = torch.arange(0, seq_length).expand(N,seq_length).to(self.device)

        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))  ## 워드 임베딩 + 포지셔널 임베딩

        for layer in layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0, device="cuda", max_length=100):
        super(Transformer,self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,heads,
            device,
            forward_expansion,
            dropout,
            max_length)

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device


    def _create_encoder(self):
        pass

    def _create_decoder(self):
        pass
