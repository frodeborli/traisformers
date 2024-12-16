# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for TrAISformer.
    https://arxiv.org/abs/2109.03958

The code is built upon:
    https://github.com/karpathy/minGPT
"""

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config, embd_size=None):
        super().__init__()
        # Use provided embd_size or fall back to config.n_embd
        self.embd_size = embd_size if embd_size is not None else config.n_embd
        assert self.embd_size % config.n_head == 0
        
        # key, query, value projections for all heads
        self.key = nn.Linear(self.embd_size, self.embd_size)
        self.query = nn.Linear(self.embd_size, self.embd_size)
        self.value = nn.Linear(self.embd_size, self.embd_size)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(self.embd_size, self.embd_size)
        # causal mask
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen, config.max_seqlen))
                                     .view(1, 1, config.max_seqlen, config.max_seqlen))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, embd_size=None):
        super().__init__()
        # Use provided embd_size or fall back to config.n_embd
        self.embd_size = embd_size if embd_size is not None else config.n_embd
        
        self.ln1 = nn.LayerNorm(self.embd_size)
        self.ln2 = nn.LayerNorm(self.embd_size)
        self.attn = CausalSelfAttention(config, self.embd_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.embd_size, 4 * self.embd_size),
            nn.GELU(),
            nn.Linear(4 * self.embd_size, self.embd_size),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TrAISformer(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config, partition_model=None):
        super().__init__()

        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.full_size = config.full_size
        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
        self.register_buffer(
            "att_sizes",
            torch.tensor([config.lat_size, config.lon_size, config.sog_size, config.cog_size]))
        self.register_buffer(
            "emb_sizes",
            torch.tensor([config.n_lat_embd, config.n_lon_embd, config.n_sog_embd, config.n_cog_embd]))

        if hasattr(config,"partition_mode"):
            self.partition_mode = config.partition_mode
        else:
            self.partition_mode = "uniform"
        self.partition_model = partition_model

        if hasattr(config,"blur"):
            self.blur = config.blur
            self.blur_learnable = config.blur_learnable
            self.blur_loss_w = config.blur_loss_w
            self.blur_n = config.blur_n
            if self.blur:
                self.blur_module = nn.Conv1d(1, 1, 3, padding=1, padding_mode='replicate', groups=1, bias=False)
                if not self.blur_learnable:
                    for params in self.blur_module.parameters():
                        params.requires_grad = False
                        params.fill_(1/3)
            else:
                self.blur_module = None

        if hasattr(config,"lat_min"): # the ROI is provided.
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max-config.lat_min
            self.lon_range = config.lon_max-config.lon_min
            self.sog_range = 30.

        if hasattr(config,"mode"): # mode: "pos" or "velo".
            self.mode = config.mode
        else:
            self.mode = "pos"

        # Passing from the 4-D space to a high-dimentional space
        self.lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer - no need to pass embd_size since it will use config.n_embd
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        if self.mode in ("mlp_pos","mlp"):
            self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        else:
            self.head = nn.Linear(config.n_embd, self.full_size, bias=False)

        self.max_seqlen = config.max_seqlen
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_max_seqlen(self):
        return self.max_seqlen

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        Configures the optimizer with appropriate weight decay settings.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add('pos_emb')   

        # Validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {str(inter_params)} in both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {str(param_dict.keys() - union_params)} unaccounted!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))],
             "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate,
                                    betas=train_config.betas)
        return optimizer


    def to_indexes(self, x, mode="uniform"):
        """Convert tokens to indexes.

        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated
                to [0,1).
            model: currenly only supports "uniform".

        Returns:
            idxs: a Tensor (dtype: Long) of indexes.
        """
        bs, seqlen, data_dim = x.shape
        if mode == "uniform":
            idxs = (x*self.att_sizes).long()
            return idxs, idxs
        elif mode in ("freq", "freq_uniform"):

            idxs = (x*self.att_sizes).long()
            idxs_uniform = idxs.clone()
            discrete_lats, discrete_lons, lat_ids, lon_ids = self.partition_model(x[:,:,:2])
#             pdb.set_trace()
            idxs[:,:,0] = torch.round(lat_ids.reshape((bs,seqlen))).long()
            idxs[:,:,1] = torch.round(lon_ids.reshape((bs,seqlen))).long()
            return idxs, idxs_uniform


    def forward(self, x, masks = None, with_targets=False, return_loss_tuple=False):
        """
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated
                to [0,1).
            masks: a Tensor of the same size of x. masks[idx] = 0. if
                x[idx] is a padding.
            with_targets: if True, inputs = x[:,:-1,:], targets = x[:,1:,:],
                otherwise inputs = x.
        Returns:
            logits, loss
        """

        if self.mode in ("mlp_pos","mlp",):
            idxs, idxs_uniform = x, x # use the real-values of x.
        else:
            # Convert to indexes
            idxs, idxs_uniform = self.to_indexes(x, mode=self.partition_mode)

        if with_targets:
            inputs = idxs[:,:-1,:].contiguous()
            targets = idxs[:,1:,:].contiguous()
            targets_uniform = idxs_uniform[:,1:,:].contiguous()
            inputs_real = x[:,:-1,:].contiguous()
            targets_real = x[:,1:,:].contiguous()
        else:
            inputs_real = x
            inputs = idxs
            targets = None

        batchsize, seqlen, _ = inputs.size()
        assert seqlen <= self.max_seqlen, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        lat_embeddings = self.lat_emb(inputs[:,:,0]) # (bs, seqlen, lat_size)
        lon_embeddings = self.lon_emb(inputs[:,:,1])
        sog_embeddings = self.sog_emb(inputs[:,:,2])
        cog_embeddings = self.cog_emb(inputs[:,:,3])
        token_embeddings = torch.cat((lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings),dim=-1)

        position_embeddings = self.pos_emb[:, :seqlen, :] # each position maps to a (learnable) vector (1, seqlen, n_embd)
        fea = self.drop(token_embeddings + position_embeddings)
        fea = self.blocks(fea)
        fea = self.ln_f(fea) # (bs, seqlen, n_embd)
        logits = self.head(fea) # (bs, seqlen, full_size) or (bs, seqlen, n_embd)

        lat_logits, lon_logits, sog_logits, cog_logits =\
            torch.split(logits, (self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)

        # Calculate the loss
        loss = None
        loss_tuple = None
        if targets is not None:

            sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_size),
                                       targets[:,:,2].view(-1),
                                       reduction="none").view(batchsize,seqlen)
            cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_size),
                                       targets[:,:,3].view(-1),
                                       reduction="none").view(batchsize,seqlen)
            lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size),
                                       targets[:,:,0].view(-1),
                                       reduction="none").view(batchsize,seqlen)
            lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size),
                                       targets[:,:,1].view(-1),
                                       reduction="none").view(batchsize,seqlen)

            if self.blur:
                lat_probs = F.softmax(lat_logits, dim=-1)
                lon_probs = F.softmax(lon_logits, dim=-1)
                sog_probs = F.softmax(sog_logits, dim=-1)
                cog_probs = F.softmax(cog_logits, dim=-1)

                for _ in range(self.blur_n):
                    blurred_lat_probs = self.blur_module(lat_probs.reshape(-1,1,self.lat_size)).reshape(lat_probs.shape)
                    blurred_lon_probs = self.blur_module(lon_probs.reshape(-1,1,self.lon_size)).reshape(lon_probs.shape)
                    blurred_sog_probs = self.blur_module(sog_probs.reshape(-1,1,self.sog_size)).reshape(sog_probs.shape)
                    blurred_cog_probs = self.blur_module(cog_probs.reshape(-1,1,self.cog_size)).reshape(cog_probs.shape)

                    blurred_lat_loss = F.nll_loss(blurred_lat_probs.view(-1, self.lat_size),
                                                  targets[:,:,0].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_lon_loss = F.nll_loss(blurred_lon_probs.view(-1, self.lon_size),
                                                  targets[:,:,1].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_sog_loss = F.nll_loss(blurred_sog_probs.view(-1, self.sog_size),
                                                  targets[:,:,2].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_cog_loss = F.nll_loss(blurred_cog_probs.view(-1, self.cog_size),
                                                  targets[:,:,3].view(-1),
                                                  reduction="none").view(batchsize,seqlen)

                    lat_loss += self.blur_loss_w*blurred_lat_loss
                    lon_loss += self.blur_loss_w*blurred_lon_loss
                    sog_loss += self.blur_loss_w*blurred_sog_loss
                    cog_loss += self.blur_loss_w*blurred_cog_loss

                    lat_probs = blurred_lat_probs
                    lon_probs = blurred_lon_probs
                    sog_probs = blurred_sog_probs
                    cog_probs = blurred_cog_probs


            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            loss = sum(loss_tuple)

            if masks is not None:
                loss = (loss*masks).sum(dim=1)/masks.sum(dim=1)

            loss = loss.mean()

        if return_loss_tuple:
            return logits, loss, loss_tuple
        else:
            return logits, loss

class TrAISformerDPE2(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Keep all original size configurations for output compatibility
        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.full_size = self.lat_size + self.lon_size + self.sog_size + self.cog_size
        
        # Register sizes for compatibility with original model
        self.register_buffer(
            "att_sizes",
            torch.tensor([config.lat_size, config.lon_size, config.sog_size, config.cog_size]))
        
        self.max_seqlen = config.max_seqlen
        self.direct_emb_size = config.direct_emb_size
        
        # Core transformer blocks
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config, self.direct_emb_size) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.direct_emb_size)
        
        # Output head - must output logits in same format as original model
        self.head = nn.Linear(self.direct_emb_size, self.full_size)

        # self.pos_projection = nn.Linear(8, self.direct_emb_size)
        self.pos_projection = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, self.direct_emb_size)
        );
        
        self.apply(self._init_weights)

    def add_positional_encoding(self, x):
        """
        Adds symmetric positional encoding to input embeddings.

        For each token at position t (1-based index):
            - Feature 1: original_feature + t
            - Feature 2: original_feature - t
            - Feature 3: original_feature + t
            - Feature 4: original_feature - t
            - ...

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_features]

        Returns:
            torch.Tensor: Projected positional encoded tensor of shape [batch_size, seq_len, direct_emb_size]
        """
        batch_size, seq_len, num_features = x.size()
        device = x.device

        # Generate position indices (1-based)
        pos_indices = torch.arange(1, seq_len + 1, device=device).float()  # [seq_len]

        # Create alternating positive and negative positional values for each feature
        # e.g., for num_features=4: [1, -1, 2, -2, 3, -3, ...]
        pos_pattern = []
        for i in range(num_features):
            t = (i // 2) + 1  # Integer division to assign positions
            sign = 1 if i % 2 == 0 else -1  # Alternate signs
            pos_pattern.append(sign)

        pos_pattern = torch.tensor(pos_pattern, device=device).float()  # [num_features]

        # Expand pos_indices to [batch_size, seq_len, num_features]
        pos_indices = pos_indices.view(1, seq_len, 1)  # [1, seq_len, 1]
        pos_indices = pos_indices.repeat(batch_size, 1, num_features)  # [batch_size, seq_len, num_features]

        # Apply the sign pattern
        pos_pattern = pos_pattern.view(1, 1, num_features)  # [1, 1, num_features]
        pos_values = pos_indices * pos_pattern  # [batch_size, seq_len, num_features]

        # Concatenate original features with positional values
        pos_encoded = torch.cat([x, pos_values], dim=-1)  # [batch_size, seq_len, 2*num_features]

        # Project the concatenated features back to direct_emb_size
        pos_encoded = self.pos_projection(pos_encoded)  # [batch_size, seq_len, direct_emb_size]

        return pos_encoded
        
    def forward(self, x, masks=None, with_targets=False, return_loss_tuple=False):
        """Forward pass with exact same interface as original model."""
        if with_targets:
            inputs = x[:, :-1, :].contiguous()
            targets = x[:, 1:, :].contiguous()
            # Convert targets to indices for loss calculation
            targets_idx = (targets * self.att_sizes).long()
        else:
            inputs = x
            targets = None
            targets_idx = None

        # Add positional encoding
        embedded = self.add_positional_encoding(inputs)
        
        # Pass through transformer
        fea = self.drop(embedded)
        fea = self.blocks(fea)
        fea = self.ln_f(fea)
        
        # Output logits with same size and meaning as original model
        logits = self.head(fea)  # Shape: [batch_size, seq_len, full_size]

        # Calculate loss exactly as original model does
        loss = None
        loss_tuple = None
        if targets_idx is not None:
            batchsize = inputs.size(0)
            seqlen = inputs.size(1)

            # Split logits into components like original model
            lat_logits, lon_logits, sog_logits, cog_logits = torch.split(
                logits,
                [self.lat_size, self.lon_size, self.sog_size, self.cog_size],
                dim=-1
            )

            sog_loss = F.cross_entropy(
                sog_logits.view(-1, self.sog_size),
                targets_idx[..., 2].view(-1),
                reduction="none"
            ).view(batchsize, seqlen)

            cog_loss = F.cross_entropy(
                cog_logits.view(-1, self.cog_size),
                targets_idx[..., 3].view(-1),
                reduction="none"
            ).view(batchsize, seqlen)

            lat_loss = F.cross_entropy(
                lat_logits.view(-1, self.lat_size),
                targets_idx[..., 0].view(-1),
                reduction="none"
            ).view(batchsize, seqlen)

            lon_loss = F.cross_entropy(
                lon_logits.view(-1, self.lon_size),
                targets_idx[..., 1].view(-1),
                reduction="none"
            ).view(batchsize, seqlen)

            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            loss = sum(loss_tuple)

            if masks is not None:
                loss = (loss * masks).sum(dim=1) / masks.sum(dim=1)
            loss = loss.mean()

        if return_loss_tuple:
            return logits, loss, loss_tuple

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {str(inter_params)} in both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {str(param_dict.keys() - union_params)} unaccounted!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], 
             "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], 
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, 
                                    betas=train_config.betas)
        return optimizer
    
    def to_indexes(self, x, mode="uniform"):
        """Convert tokens to indexes - identical to original model."""
        bs, seqlen, data_dim = x.shape
        idxs = (x * self.att_sizes).long()
        return idxs, idxs
    
    def get_max_seqlen(self):
        """Added for compatibility with original TrAISformer"""
        return self.max_seqlen
    
class TrAISformerDPE(TrAISformer):
    def __init__(self, config):
        super().__init__(config)
        # Modify or replace the existing positional embeddings
        # If we want a simpler positional encoding, we could use a simple learned positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        # You could add alternative positional encoding schemes here, like sinusoidal or any custom method

    def forward(self, x, masks=None, with_targets=False, return_loss_tuple=False):
        # Convert to indexes or use real values based on the encoding strategy
        idxs, idxs_uniform = self.to_indexes(x, mode=self.partition_mode) if not hasattr(self.config, 'use_raw_positions') else (x, x)

        if with_targets:
            inputs = idxs[:, :-1, :].contiguous()
            targets = idxs[:, 1:, :].contiguous()
            targets_uniform = idxs_uniform[:, 1:, :].contiguous()
            inputs_real = x[:, :-1, :].contiguous()
            targets_real = x[:, 1:, :].contiguous()
        else:
            inputs_real = x
            inputs = idxs
            targets = None

        batchsize, seqlen, _ = inputs.size()
        assert seqlen <= self.max_seqlen, "Cannot forward, model block size is exhausted."

        # Embedding layers for lat, lon, sog, cog
        lat_embeddings = self.lat_emb(inputs[:, :, 0])  # (bs, seqlen, n_lat_embd)
        lon_embeddings = self.lon_emb(inputs[:, :, 1])
        sog_embeddings = self.sog_emb(inputs[:, :, 2])
        cog_embeddings = self.cog_emb(inputs[:, :, 3])

        # Concatenate the embeddings
        token_embeddings = torch.cat((lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings), dim=-1)

        # Positional embeddings (different from original TrAISformer)
        position_embeddings = self.pos_emb[:, :seqlen, :]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)

        # Pass through the transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)  # Layer normalization at the output
        logits = self.head(x)

        # Loss computation if targets are provided
        if targets is not None:
            logits, loss, loss_tuple = self.compute_loss(logits, targets, masks)
            if return_loss_tuple:
                return logits, loss, loss_tuple
            return logits, loss
        return logits

    def compute_loss(self, logits, targets, masks):
        # Compute losses specific to the task; implement as needed based on the logits and targets
        # Assume some loss function here; implement it based on your requirements
        loss = None
        # Compute loss as per the specific needs; this is just a placeholder
        return logits, loss, None  # Modify as necessary to compute actual loss components

# Configuration for TrAISformer must be defined or imported before instantiating
# config = SomeConfigClass()
# model = TrAISformerDPE(config)
