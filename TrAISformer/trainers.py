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

"""Boilerplate for training a neural network.

References:
    https://github.com/karpathy/minGPT
"""

import os
import math
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils
from models import TrAISformer, TrAISformerDPE
from trAISformer import TB_LOG, tb

logger = logging.getLogger(__name__)


@torch.no_grad()
def sample(model, seqs, steps, temperature=1.0, sample=False, sample_mode="pos_vicinity", 
          r_vicinity=20, top_k=None):
    """
    Take a conditioning sequence of AIS observations seq and predict the next observation,
    feed the predictions back into the model each time. Works with both original TrAISformer 
    and DPE model.
    """
    # Handle both model types for max_seqlen
    max_seqlen = model.max_seqlen if hasattr(model, 'max_seqlen') else model.get_max_seqlen()
    model.eval()
    
    for k in range(steps):
        # Crop context if needed
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]
        
        # Get predictions
        logits, _ = model(seqs_cond)
        
        d2inf_pred = torch.zeros((logits.shape[0], 4)).to(seqs.device) + 0.5
        
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        
        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)
            
        if sample_mode in ("pos_vicinity",):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])
            lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = utils.top_k_logits(lat_logits, top_k)
            lon_logits = utils.top_k_logits(lon_logits, top_k)
            sog_logits = utils.top_k_logits(sog_logits, top_k)
            cog_logits = utils.top_k_logits(cog_logits, top_k)

        # apply softmax to convert to probabilities
        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)

        # sample from the distribution or take the most likely
        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        # convert to x (range: [0,1))
        next_token = (ix.float() + d2inf_pred) / model.att_sizes
            
        # Append prediction to sequence
        seqs = torch.cat((seqs, next_token.unsqueeze(1)), dim=1)

    return seqs

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config, savedir=None, 
                 device=torch.device("cpu"), aisdls={}, INIT_SEQLEN=0):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir
        self.device = device
        self.model = model.to(device)
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN
        
        # Initialize optimizer
        raw_model = model.module if hasattr(model, "module") else model
        self.optimizer = raw_model.configure_optimizers(config)
        
        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        self.tokens = 0  # counter used for learning rate decay

    def run_epoch(self, split, epoch=0):
        """Run one epoch of training or evaluation."""
        is_train = split == 'Training'
        model = self.model
        config = self.config
        
        # Set model mode
        model.train(is_train)
        data = self.train_dataset if is_train else self.test_dataset

        # Optimized DataLoader configuration
        loader = DataLoader(
            data,
            shuffle=True if is_train else False,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
            drop_last=is_train
        )

        # Initialize tracking variables
        losses = []
        n_batches = len(loader)
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        d_loss, d_reg_loss, d_n = 0, 0, 0

        # Check device type for mixed precision
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            # Efficient data transfer to GPU
            seqs = seqs.to(self.device, non_blocking=True)
            masks = masks[:, :-1].to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with torch.set_grad_enabled(is_train):
                with torch.amp.autocast(device_type=device_type, enabled=config.mixed_precision):
                    # Handle both positional encoding types
                    if hasattr(model, 'pos_encoding') and model.pos_encoding == "direct":
                        logits, loss = model(seqs, masks=masks, with_targets=True)
                    else:
                        # Original learned positional encoding path
                        logits, loss = model(seqs, masks=masks, with_targets=True)
                    
                    loss = loss.mean()

            # Accumulate loss statistics
            losses.append(loss.item())
            d_loss += loss.item() * seqs.shape[0]
            d_n += seqs.shape[0]

            if is_train:
                # Optimized training step
                model.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                if config.mixed_precision and device_type == 'cuda':
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    self.optimizer.step()

                # Learning rate decay handling
                if config.lr_decay:
                    self.tokens += (seqs >= 0).sum()
                    if self.tokens < config.warmup_tokens:
                        # Linear warmup
                        lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # Cosine learning rate decay
                        progress = float(self.tokens - config.warmup_tokens) / float(
                            max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                # Update progress bar
                pbar.set_description(
                    f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}"
                )

        # End of epoch logging
        if is_train:
            logging.info(
                f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, lr {lr:e}."
            )
        else:
            logging.info(
                f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}."
            )
            test_loss = float(np.mean(losses))
            return test_loss

    def train(self):
        """Main training loop."""
        best_loss = float('inf')
        best_epoch = 0

        for epoch in range(self.config.max_epochs):
            # Training phase
            self.run_epoch('Training', epoch)
            
            # Validation phase
            if self.test_dataset is not None:
                test_loss = self.run_epoch('Valid', epoch)

                # Save checkpoint if model improved
                good_model = test_loss < best_loss
                if self.config.ckpt_path is not None and good_model:
                    best_loss = test_loss
                    best_epoch = epoch
                    self.save_checkpoint(best_epoch + 1)

            # Plot sample predictions
            self._plot_samples(epoch)

    def save_checkpoint(self, best_epoch):
        """Save model checkpoint."""
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def _plot_samples(self, epoch):
        """Plot sample predictions for visualization."""
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        seqs, masks, seqlens, mmsis, time_starts = next(iter(self.aisdls["test"]))
        n_plots = 7
        init_seqlen = self.INIT_SEQLEN
        seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)
        
        preds = sample(raw_model,
                      seqs_init,
                      96 - init_seqlen,
                      temperature=1.0,
                      sample=True,
                      sample_mode=self.config.sample_mode,
                      r_vicinity=self.config.r_vicinity,
                      top_k=self.config.top_k)

        # Create visualization
        img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')
        plt.figure(figsize=(9, 6), dpi=150)
        cmap = plt.cm.get_cmap("jet")
        preds_np = preds.detach().cpu().numpy()
        inputs_np = seqs.detach().cpu().numpy()
        
        for idx in range(n_plots):
            c = cmap(float(idx) / (n_plots))
            try:
                seqlen = seqlens[idx].item()
            except:
                continue
            plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], color=c)
            plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], "o", markersize=3, color=c)
            plt.plot(inputs_np[idx][:seqlen, 1], inputs_np[idx][:seqlen, 0], linestyle="-.", color=c)
            plt.plot(preds_np[idx][init_seqlen:, 1], preds_np[idx][init_seqlen:, 0], "x", markersize=4, color=c)
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.savefig(img_path, dpi=150)
        plt.close()

class Trainer2:

    def __init__(self, model, train_dataset, test_dataset, config, savedir=None, device=torch.device("cpu"), aisdls={}, INIT_SEQLEN=0):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir
        self.device = device
        self.model = model.to(device)
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN
        # Add gradient scaler for mixed precision training
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = torch.amp.GradScaler(enabled=config.mixed_precision and device_type == 'cuda')

    def save_checkpoint(self, best_epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config, aisdls, INIT_SEQLEN = self.model, self.config, self.aisdls, self.INIT_SEQLEN
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        
        if model.mode in ("gridcont_gridsin", "gridcont_gridsigmoid", "gridcont2_gridsigmoid"):
            return_loss_tuple = True
        else:
            return_loss_tuple = False


        def run_epoch(self, split, epoch=0):
            """Run one epoch of training or evaluation.
            
            Args:
                split (str): 'Training' or 'Valid'
                epoch (int): Current epoch number
                
            Returns:
                float: Test loss if split=='Valid', None otherwise
            """
            is_train = split == 'Training'
            model = self.model
            config = self.config
            
            # Set model mode
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset

            # Optimized DataLoader configuration
            loader = DataLoader(
                data,
                shuffle=True if is_train else False,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=config.persistent_workers,
                prefetch_factor=config.prefetch_factor,
                drop_last=is_train
            )

            # Initialize tracking variables
            losses = []
            n_batches = len(loader)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_reg_loss, d_n = 0, 0, 0

            # Check device type for mixed precision
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
                # Efficient data transfer to GPU
                seqs = seqs.to(self.device, non_blocking=True)
                masks = masks[:, :-1].to(self.device, non_blocking=True)

                # Forward pass with mixed precision
                with torch.set_grad_enabled(is_train):
                    with torch.amp.autocast(device_type=device_type, enabled=config.mixed_precision):
                        # Handle both positional encoding types
                        if hasattr(model, 'pos_encoding') and model.pos_encoding == "direct":
                            if return_loss_tuple:
                                logits, loss, loss_tuple = model(
                                    seqs, 
                                    masks=masks,
                                    with_targets=True,
                                    return_loss_tuple=True
                                )
                            else:
                                logits, loss = model(
                                    seqs,
                                    masks=masks,
                                    with_targets=True
                                )
                        else:
                            # Original learned positional encoding path
                            if return_loss_tuple:
                                logits, loss, loss_tuple = model(
                                    seqs,
                                    masks=masks,
                                    with_targets=True,
                                    return_loss_tuple=True
                                )
                            else:
                                logits, loss = model(
                                    seqs,
                                    masks=masks,
                                    with_targets=True
                                )
                        
                        loss = loss.mean()

                # Accumulate loss statistics
                losses.append(loss.item())
                d_loss += loss.item() * seqs.shape[0]
                if return_loss_tuple:
                    reg_loss = loss_tuple[-1]
                    reg_loss = reg_loss.mean()
                    d_reg_loss += reg_loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]

                if is_train:
                    # Optimized training step
                    model.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                    
                    if config.mixed_precision and device_type == 'cuda':
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        self.optimizer.step()

                    # Learning rate decay handling
                    if config.lr_decay:
                        self.tokens += (seqs >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            # Linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # Cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # Update progress bar
                    pbar.set_description(
                        f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}"
                    )

                    # Tensorboard logging if enabled
                    if TB_LOG:
                        tb.add_scalar("loss", loss.item(), epoch * n_batches + it)
                        tb.add_scalar("lr", lr, epoch * n_batches + it)
                        for name, params in model.head.named_parameters():
                            tb.add_histogram(f"head.{name}", params, epoch * n_batches + it)
                            tb.add_histogram(f"head.{name}.grad", params.grad, epoch * n_batches + it)
                        if hasattr(model, 'res_pred'):
                            for name, params in model.res_pred.named_parameters():
                                tb.add_histogram(f"res_pred.{name}", params, epoch * n_batches + it)
                                tb.add_histogram(f"res_pred.{name}.grad", params.grad, epoch * n_batches + it)

            # End of epoch logging
            if is_train:
                if return_loss_tuple:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, "
                        f"{d_reg_loss / d_n:.5f}, lr {lr:e}."
                    )
                else:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, lr {lr:e}."
                    )
            else:
                if return_loss_tuple:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}."
                    )
                else:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}."
                    )

                test_loss = float(np.mean(losses))
                return test_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        best_epoch = 0

        for epoch in range(config.max_epochs):

            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                best_epoch = epoch
                self.save_checkpoint(best_epoch + 1)

            ## SAMPLE AND PLOT
            # ==========================================================================================
            # ==========================================================================================
            raw_model = model.module if hasattr(self.model, "module") else model
            seqs, masks, seqlens, mmsis, time_starts = next(iter(aisdls["test"]))
            n_plots = 7
            init_seqlen = INIT_SEQLEN
            seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)
            preds = sample(raw_model,
                           seqs_init,
                           96 - init_seqlen,
                           temperature=1.0,
                           sample=True,
                           sample_mode=self.config.sample_mode,
                           r_vicinity=self.config.r_vicinity,
                           top_k=self.config.top_k)

            img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')
            plt.figure(figsize=(9, 6), dpi=150)
            cmap = plt.cm.get_cmap("jet")
            preds_np = preds.detach().cpu().numpy()
            inputs_np = seqs.detach().cpu().numpy()
            for idx in range(n_plots):
                c = cmap(float(idx) / (n_plots))
                try:
                    seqlen = seqlens[idx].item()
                except:
                    continue
                plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], color=c)
                plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], "o", markersize=3, color=c)
                plt.plot(inputs_np[idx][:seqlen, 1], inputs_np[idx][:seqlen, 0], linestyle="-.", color=c)
                plt.plot(preds_np[idx][init_seqlen:, 1], preds_np[idx][init_seqlen:, 0], "x", markersize=4, color=c)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.savefig(img_path, dpi=150)
            plt.close()

        # Final state
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Last epoch: {epoch:03d}, saving model to {self.config.ckpt_path}")
        save_path = self.config.ckpt_path.replace("model.pt", f"model_{epoch + 1:03d}.pt")
        torch.save(raw_model.state_dict(), save_path)
