import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

class DM(nn.Module):
    def __init__(self, modeltype, nfeats, translation, latent_dim=256, ff_size=1024,
            num_layers=8, num_heads=4, dropout=0.1, activation="gelu", cond_feat ="image",
            dataset='fixation', arch='trans_enc', **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.nfeats = nfeats

        self.dataset = dataset

        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.cond_feat = cond_feat

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.nfeats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.avg_pooling = nn.AvgPool1d(8)

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.AvgPool1d(8),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(3520, 1024),

            # Semantic Feature Ablation
            # nn.Linear(2560, 1024),

            # Dynamic Feature Ablation
            # nn.Linear(1600, 1024),
            
            nn.ReLU(),
            nn.AvgPool1d(2),
        )

        self.fc3 = nn.Linear(25, self.latent_dim)

        self.output_process = OutputProcess(self.nfeats, self.latent_dim)

    def parameters(self):
        return [p for name, p in self.named_parameters()]

    def mask_cond(self, cond, force_mask=False):
        _, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        if self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def emb_cond(self, cond):
        if self.cond_feat == "image":
            # Semantic Feature Ablation
            # index = [2,9,10]
            
            # Dynamic Feature Ablation          
            # index = [3,4,5,6,7,8]

            # device = torch.device('cuda')
            # mask = torch.ones(11,dtype=torch.bool).to(device)
            # for idx in index:
            #     mask[idx] = False
            # cond = cond.index_select(2, mask.nonzero().squeeze())
            # print(cond.shape)

            batch_size, framePerData, feature_count, resnet_output_dim = cond.shape
            cond = cond.view(batch_size, framePerData*feature_count, resnet_output_dim)
            cond = self.fc1(cond)
            cond = cond.view(batch_size, framePerData*feature_count*64)
            cond = cond.unsqueeze(0)
            cond = self.fc2(cond)
        elif self.cond_feat == "fixhead":
            cond = cond.to(torch.float32)
            batch_size, framePerData, feature_count = cond.shape
            cond = cond.view(batch_size, framePerData*feature_count)
            cond = cond.unsqueeze(0)
            cond = self.fc3(cond)
        return cond

    def forward(self, x, timesteps, y=None):
        emb = self.embed_timestep(timesteps)
        device = torch.device('cuda')
        cond = torch.tensor(y['cond']).to(device)
        enc_cond = self.emb_cond(cond)
        enc_cond = self.mask_cond(enc_cond)
        emb += enc_cond
        x = self.input_process(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq)[1:]
        output = self.output_process(output)
        return output


    def _apply(self, fn):
        super()._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)


class InputProcess(nn.Module):
    def __init__(self, nfeats, latent_dim):
        super().__init__()
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.nfeats, self.latent_dim)
        
    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        x = self.poseEmbedding(x)
        return x
    

class OutputProcess(nn.Module):
    def __init__(self, nfeats, latent_dim):
        super().__init__()
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.nfeats)

    def forward(self, output):
        output = self.poseFinal(output)
        nframes, bs, d = output.shape
        output = output.reshape(nframes, bs, d, 1)
        output = output.permute(1, 2, 3, 0) 
        return output
    
