#!/usr/bin/env python
# coding: utf-8
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


#
# This file contains a modified version of the original source code.
# Some functionality changes have been made to the original code, available from the CLI,
# but mainly the code has been refactored and put into functions in "main_functions.py"
#


import models, trainers, utils
from config_trAISformer import Config
from main_functions import setup_logging, load_datasets, evaluate_model, plot_errors, make_prediction
import pickle
import argparse
import numpy as np
from tqdm import tqdm

cf = Config()
TB_LOG = cf.tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--true", action="store_true", help="Dump true trajectories of test set to 'output/true.pkl'")
    parser.add_argument("--evalInitSize", type=int, help="Length of track to make predictions on", default=cf.init_seqlen)
    parser.add_argument("--filename", type=str, help="Name add-on to prediction error PNG", default="")
    parser.add_argument("--predict", action="store_true", help="Make prediction(s)")
    parser.add_argument("--numSteps", type=int, help="Number of steps to predict ahead for", default=100)
    parser.add_argument("--nPreds", type=int, help="Number of predictions to make for each track", default=16)
    parser.add_argument("--initLen", type=int, help="Initial length of the track we predict for", default=cf.init_seqlen)
    parser.add_argument("--num", type=int, help="Number of tracks to predict for (default: all)", default=0)
    parser.add_argument("--pos_encoding", type=str, default="learned")
    parser.add_argument("--direct_emb_size", type=int, default=128)
    args = parser.parse_args()

    if args.train:
        cf.retrain = True
    if args.evaluate:
        cf.evaluate = True

    # Add new config parameters from arguments
    if args.pos_encoding:
        cf.pos_encoding = args.pos_encoding
    if args.direct_emb_size:
        cf.direct_emb_size = args.direct_emb_size

    setup_logging(cf)

    # make deterministic
    utils.set_seed(42)

    # Dump true trajectories of test set to output/true.pkl
    if args.true:
        with open("data/ais/ais_test.pkl", "rb") as f:
            tracks = pickle.load(f)
        with open("output/true.pkl", "wb") as f:
            pickle.dump(tracks, f)
        print("[+] Saved true trajectories to 'output/true.pkl'")
        exit()

    # Make predictions with TrAISformer model
    if args.predict:
        if cf.pos_encoding == "direct":
            model = models.TrAISformerDPE(cf).to(cf.device)
        else:
            model = models.TrAISformer(cf, partition_model=None).to(cf.device)

        with open("data/ais/ais_test.pkl", "rb") as f:
            tracks = pickle.load(f)
        if args.num > 0:
            tracks = tracks[:args.num]

        preds = []
        correct, fail, error = 0, 0, 0
        # Predict trajectories
        for track in tqdm(tracks, desc="Predicting trajectories", total=len(tracks)):
            track = np.array([track["traj"][:args.initLen, :4]])
            res = make_prediction(track=track, num_steps=args.numSteps, n_preds=args.nPreds, model=model)
            preds.append(res)

        # Store predictions in output/ folder
        filename = f"output/{args.initLen}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(preds, f)
        print(f"[+] Saved predictions to {filename}")

    else:
        # Load datasets, model and trainer
        aisdatasets, aisdls = load_datasets(cf)

        if cf.pos_encoding == "direct":
            model = models.TrAISformerDPE(cf).to(cf.device)
        else:
            model = models.TrAISformer(cf, partition_model=None).to(cf.device)

        trainer = trainers.Trainer(
                model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, aisdls=aisdls, INIT_SEQLEN=cf.init_seqlen
        )
        # Train the model
        if cf.retrain:
            if args.initLen:
                cf.init_seqlen = args.initLen
            trainer.train()

        # Evaluate the model
        if cf.evaluate:
            cf.init_seqlen = args.evalInitSize if args.evalInitSize else cf.init_seqlen
            pred_errors = evaluate_model(cf, aisdls, model)
            plot_errors(cf, pred_errors, f"_{args.filename}" if args.filename else "")
