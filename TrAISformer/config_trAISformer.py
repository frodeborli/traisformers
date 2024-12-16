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

"""Configuration flags to run the main script.
"""

import os
import torch
os.environ['NUMEXPR_MAX_THREADS'] = '32'

class Config():
    pos_encoding = "learned"  # "learned" or "direct"
    direct_emb_size = 128    # Size for direct encoding
    retrain = False
    evaluate = False
    tb_log = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    max_epochs = 50
    batch_size = 32
    n_samples = 16

    init_seqlen = 12
    max_seqlen = 120
    min_seqlen = 36

    dataset_name = "ais"

    # Add new optimization settings
    mixed_precision = True
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 2
    gradient_checkpointing = True


    if dataset_name == "ct_dma": #==============================

        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and
        # dlon
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72

        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128

        lat_min = 55.5
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13
    else:
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72

        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128

        lat_min = 69.2
        lat_max = 73.0
        lon_min = 13.0
        lon_max = 31.5


    #===========================================================================
    # Model and sampling flags
    mode = "pos"  #"pos", "pos_grad", "mlp_pos", "mlpgrid_pos", "velo", "grid_l2", "grid_l1",
                            # "ce_vicinity", "gridcont_grid", "gridcont_real", "gridcont_gridsin", "gridcont_gridsigmoid"
    sample_mode =  "pos_vicinity" # "pos", "pos_vicinity" or "velo"
    top_k = 10 # int or None
    r_vicinity = 40 # int

    # Blur flags
    #===================================================
    blur = True
    blur_learnable = False
    blur_loss_w = 1.0
    blur_n = 2
    if not blur:
        blur_n = 0
        blur_loss_w = 0

    # Data flags
    #===================================================
    datadir = f"./data/{dataset_name}/"
    trainset_name = f"{dataset_name}_train.pkl"
    validset_name = f"{dataset_name}_valid.pkl"
    testset_name = f"{dataset_name}_test.pkl"


    # model parameters
    #===================================================
    n_head = 8
    n_layer = 8
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
    # base GPT config, params common to all GPT versions
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    # optimization parameters
    #===================================================
    learning_rate = 6e-4 # 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = True
    warmup_tokens = 512*20 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    num_workers = 16 # for DataLoader

    def __init__(self):
        self.update_paths()

    def update_paths(self):
        """Update filename and paths based on current configuration"""
        prefix = "DPE-" if self.pos_encoding == "direct" else ""
        self.filename = f"{prefix}{self.dataset_name}"\
            + f"-{self.mode}-{self.sample_mode}-{self.top_k}-{self.r_vicinity}"\
            + f"-blur-{self.blur}-{self.blur_learnable}-{self.blur_n}-{self.blur_loss_w}"\
            + f"-data_size-{self.lat_size}-{self.lon_size}-{self.sog_size}-{self.cog_size}"\
            + f"-embd_size-{self.n_lat_embd}-{self.n_lon_embd}-{self.n_sog_embd}-{self.n_cog_embd}"\
            + f"-head-{self.n_head}-{self.n_layer}"\
            + f"-bs-{self.batch_size}"\
            + f"-lr-{self.learning_rate}"\
            + f"-seqlen-{self.init_seqlen}-{self.max_seqlen}"
        self.savedir = "./results/"+self.filename+"/"
        self.ckpt_path = os.path.join(self.savedir, "model.pt")