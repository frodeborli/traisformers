import matplotlib.pyplot as plt
import numpy as np
import utils
import torch
import os
import datasets
import pickle
import trainers
from tqdm import tqdm
from torch.utils.data import DataLoader
from config_trAISformer import Config
from models import TrAISformer

"""
This file contains code from the original trAISformer.py file,
but refactored into functions
"""

def setup_logging(cf: Config) -> None:
    """
    Create or use existing logs
    """
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils.new_log(cf.savedir, "log")

def movement_filter(ais_tracks: list) -> list:
    """
    Filter out AIS tracks that move too short between messages
    """
    moving_threshold = 0.016 # Min-speed ~0.5
    for V in ais_tracks:
        try:
            moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
        except:
            moving_idx = len(V["traj"]) - 1
        V["traj"] = V["traj"][moving_idx:, :]
    return ais_tracks

def create_pytorch_datasets(cf: Config, Data: dict, phase: str, aisdatasets: dict, aisdls: dict) -> tuple[dict, dict]:
    """
    Create pytorch datasets of AIS tracks
    """
    if cf.mode in ("pos_grad", "grad"):
        aisdatasets[phase] = datasets.AISDataset_grad(Data[phase], max_seqlen=cf.max_seqlen + 1, device=cf.device)
    else:
        aisdatasets[phase] = datasets.AISDataset(Data[phase], max_seqlen=cf.max_seqlen + 1, device=cf.device)
    if phase == "test":
        shuffle = False
    else:
        shuffle = True
    aisdls[phase] = DataLoader(aisdatasets[phase], batch_size=cf.batch_size, shuffle=shuffle)
    return aisdatasets, aisdls


def load_datasets(cf: Config) -> tuple[dict, dict]:
    """
    Load training, validation and test datasets
    into pytorch format
    """
    Data, aisdatasets, aisdls = {}, {}, {}
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            ais_tracks = pickle.load(f)

        ais_tracks = movement_filter(ais_tracks)
        Data[phase] = [x for x in ais_tracks if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]

        print(len(ais_tracks), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset...")
        aisdatasets, aisdls = create_pytorch_datasets(cf, Data, phase, aisdatasets, aisdls)
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen
    return aisdatasets, aisdls

def plot_errors(cf: Config, pred_errors: list, add2filename: str="") -> None:
    """
    Plot figures visualizing the trajectory errors
    """
    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors)) / 12
    plt.plot(v_times, pred_errors)

    for i in range(1,4):
        timestep = i * 12
        plt.plot(i, pred_errors[timestep], "o")
        plt.plot([i, i], [0, pred_errors[timestep]], "r")
        plt.plot([0, i], [pred_errors[timestep], pred_errors[timestep]], "r")
        plt.text(i + 0.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    plt.xlim([0, 8])
    plt.ylim([0,pred_errors.max()+1])
    plt.savefig(cf.savedir + f"prediction_error{add2filename}.png")

def apply_masks(l_val: list, l_masks: list) -> np.ndarray:
    """
    Apply masks to the errors based on haversine distance
    """
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_val, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    return pred_errors.detach().cpu().numpy()

def evaluate_model(cf: Config, aisdls: dict, model: TrAISformer) -> np.ndarray:
    """
    Evaluates the performance of the trained model
    """
    model.load_state_dict(torch.load(cf.ckpt_path, map_location=cf.device))
    model.eval()

    l_min_errors, l_mean_errors, l_masks = [], [], []
    v_ranges = torch.tensor([2, 3, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)
    max_seqlen = cf.init_seqlen + 12*6
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))

    with torch.no_grad():
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            seqs_init = seqs[:, :cf.init_seqlen, :].to(cf.device)
            masks = masks[:, :max_seqlen].to(cf.device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - cf.init_seqlen, cf.n_samples)).to(cf.device)
            error_ens, _ = eval_predict(cf, model, seqs_init, max_seqlen, seqs, v_ranges, v_roi_min, masks, error_ens)

            # Accumulation through batches
            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, cf.init_seqlen:])

    #l_min = [x.values for x in l_min_errors]
    l_min = l_mean_errors
    return apply_masks(l_min, l_masks)

def eval_predict(cf: Config, model: TrAISformer, seqs_init: torch.Tensor, max_seqlen: int,
                 seqs: torch.Tensor, v_ranges: torch.Tensor, v_roi_min: torch.Tensor,
                 masks: torch.Tensor, error_ens: torch.Tensor) -> tuple[torch.Tensor, list]:
    """
    Makes N predictions on the input track, and returns a tuple with:
        - pytorch Tensor with errors
        - list with tuples of input coordinates and predicted coordinates (input_coords, pred_coords)
    """
    torch.pi = torch.acos(torch.zeros(1)).item() * 2
    all_preds = []
    for i_sample in range(cf.n_samples):
        preds = trainers.sample(model, seqs_init, max_seqlen - cf.init_seqlen,
                                temperature=1.0, sample=True, sample_mode=cf.sample_mode,
                                r_vicinity=cf.r_vicinity, top_k=cf.top_k)
        inputs = seqs[:, :max_seqlen, :].to(cf.device)
        input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
        pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
        all_preds.append((input_coords.cpu().detach().numpy(), pred_coords.cpu().detach().numpy()))

        d = utils.haversine(input_coords, pred_coords) * masks
        error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]
    return error_ens, all_preds


def predict(cf: Config, model: TrAISformer, seq: torch.Tensor, steps: int, n_preds: int) -> list:
    """
    Predicts n_preds trajectories for the given input.
    The predictions are appended to a list and returned
    """
    model.load_state_dict(torch.load(cf.ckpt_path, map_location=cf.device))
    model.eval()

    all_preds = []
    with torch.no_grad():
        seqs_init = seq.to(cf.device)
        for i_sample in range(n_preds):
            preds = trainers.sample(model, seqs_init, steps, temperature=1.0, sample=True, sample_mode=cf.sample_mode, r_vicinity=cf.r_vicinity, top_k=cf.top_k)
            all_preds.append(preds.cpu().detach().numpy())
    return all_preds

def make_prediction(track: list, num_steps: int=20, n_preds: int=3, cf: Config=None, model: TrAISformer=None) -> list:
    """
    Make n_preds predictions for the given track(s), predicting num_steps messages ahead

    Parameters:
    track (list): input track to predict trajectories for
    num_steps (int): number of steps to predict ahead
    n_preds (int): number of predictions to make

    Returns:
    list: list of trajectory predictions for the given track
    """
    if cf is None:
        cf = Config()
    if model is None:
        model = TrAISformer(cf, partition_model=None).to(cf.device)
    # Predict for the given input track
    track = torch.Tensor(track)
    res = predict(cf, model, seq=track, steps=num_steps, n_preds=n_preds)
    return res
