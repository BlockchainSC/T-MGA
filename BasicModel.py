#!/usr/bin/env/python
from __future__ import print_function
from typing import List, Any, Sequence
from utils import MLP, ThreadedIterator

import tensorflow as tf


import time
import os
import json
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe 
from matplotlib.patches import Patch


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 2.2
})
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

print(">>> FILE LOADED: BasicModel.py", flush=True)


def smooth_curve(y, window_size=10):
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')

#  YAHAN PASTE KARO (helper functions)
def save_histories(out_path, train_loss, valid_loss, train_acc, valid_acc):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_acc": train_acc,
        "valid_acc": valid_acc
    }
    with open(out_path, "w") as f:
        json.dump(data, f)

def load_histories(in_path):
    with open(in_path, "r") as f:
        data = json.load(f)
    return data["train_loss"], data["valid_loss"], data["train_acc"], data["valid_acc"]

def save_metrics(out_path, metrics):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

def load_metrics(in_path):
    with open(in_path, "r") as f:
        return json.load(f)

def infer_dataset_name(train_file):
    lower = train_file.lower()
    if "reentrancy" in lower:
        return "reentrancy"
    if "timestamp" in lower:
        return "timestamp"
    if "integeroverflow" in lower:
        return "integeroverflow"
    return "run"

def compute_eval_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = 0.0
    return metrics

def plot_accuracy_loss_curves(re_hist_path, ts_hist_path, out_dir="./result", max_epochs=None):
    if not (os.path.exists(re_hist_path) and os.path.exists(ts_hist_path)):
        print("Skipping accuracy/loss curves: history files missing.")
        return

    re_train_loss, re_valid_loss, re_train_acc, re_valid_acc = load_histories(re_hist_path)
    ts_train_loss, ts_valid_loss, ts_train_acc, ts_valid_acc = load_histories(ts_hist_path)

    min_len = min(len(re_train_acc), len(ts_train_acc))
    if max_epochs is not None:
        min_len = min(min_len, int(max_epochs))
    epochs = np.arange(1, min_len + 1)

    re_train_acc = re_train_acc[:min_len]
    re_valid_acc = re_valid_acc[:min_len]
    ts_train_acc = ts_train_acc[:min_len]
    ts_valid_acc = ts_valid_acc[:min_len]
  
    re_train_loss = re_train_loss[:min_len]
    re_valid_loss = re_valid_loss[:min_len]
    ts_train_loss = ts_train_loss[:min_len]
    ts_valid_loss = ts_valid_loss[:min_len]

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Accuracy: Train = GREEN, Valid = ORANGE (no markers)
    plt.plot(epochs, np.array(re_train_acc) * 100.0, label="Reentrancy Train",
            color="green", linestyle="-", linewidth=2)

    plt.plot(epochs, np.array(re_valid_acc) * 100.0, label="Reentrancy Valid",
            color="orange", linestyle="-", linewidth=2)

    plt.plot(epochs, np.array(ts_train_acc) * 100.0, label="Timestamp Train",
            color="green", linestyle="-", linewidth=2)

    line_ts_valid, = plt.plot(epochs, np.array(ts_valid_acc) * 100.0, label="Timestamp Valid",
                            color="orange", linestyle="-", linewidth=2)

    # Optional: orange line ko clear banane ke liye outline
    line_ts_valid.set_path_effects([
        pe.Stroke(linewidth=4.0, foreground="black"),
        pe.Normal()
    ])

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Accuracy over Epochs", fontsize=14)
    plt.legend(frameon=True)
    plt.grid(True, color="0.7", linewidth=0.8)
    plt.tight_layout()
    
    plt.savefig(os.path.join(out_dir, "acc_curve.png"), dpi=150)
    plt.savefig(os.path.join(out_dir, "acc_curve.pdf"))
    plt.close()
    # ---- LOSS ----
    # Loss: Train = BLUE, Valid = RED (no markers, same linestyle)
    plt.figure(figsize=(10, 6))

    # log scale safe (MUST be before plotting)
    eps = 1e-8
    re_train_loss_s = np.maximum(np.array(re_train_loss, dtype=float), eps)
    re_valid_loss_s = np.maximum(np.array(re_valid_loss, dtype=float), eps)
    ts_train_loss_s = np.maximum(np.array(ts_train_loss, dtype=float), eps)
    ts_valid_loss_s = np.maximum(np.array(ts_valid_loss, dtype=float), eps)

    plt.plot(epochs, re_train_loss_s, label="Reentrancy Train",
            color="blue", linestyle="-", linewidth=2)
    plt.plot(epochs, re_valid_loss_s, label="Reentrancy Valid",
            color="red", linestyle="-", linewidth=2)
    plt.plot(epochs, ts_train_loss_s, label="Timestamp Train",
            color="blue", linestyle="-", linewidth=2)
    plt.plot(epochs, ts_valid_loss_s, label="Timestamp Valid",
            color="red", linestyle="-", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss over Epochs", fontsize=14)
    plt.legend(frameon=True)
    plt.grid(True, color="0.7", linewidth=0.8)
    plt.tight_layout()

    plt.yscale("log")  # safe now

    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.savefig(os.path.join(out_dir, "loss_curve.pdf"))
    plt.close()


  


def plot_final_accuracy_bars(re_hist_path, ts_hist_path, out_dir="./result"):
    if not (os.path.exists(re_hist_path) and os.path.exists(ts_hist_path)):
        print("Skipping final accuracy bars: history files missing.")
        return

    re_train_loss, re_valid_loss, re_train_acc, re_valid_acc = load_histories(re_hist_path)
    ts_train_loss, ts_valid_loss, ts_train_acc, ts_valid_acc = load_histories(ts_hist_path)

    re_train_final = float(re_train_acc[-1]) * 100.0
    re_valid_final = float(re_valid_acc[-1]) * 100.0
    ts_train_final = float(ts_train_acc[-1]) * 100.0
    ts_valid_final = float(ts_valid_acc[-1]) * 100.0

    labels = ["Reentrancy", "Timestamp"]
    train_vals = [re_train_final, ts_train_final]
    valid_vals = [re_valid_final, ts_valid_final]

    x = np.arange(len(labels))
    width = 0.12

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))

    # ONLY TWO COLORS (train=blue, valid=orange)
    train_color = "#1f77b4"
    valid_color = "#ff7f0e"

    # hatches same
    train_hatches = ["///", "xx"]
    valid_hatches = ["\\\\", ".."]

    edge = "black"
    plt.rcParams["hatch.linewidth"] = 0.8

    for i in range(len(labels)):
        plt.bar(x[i] - width/2, train_vals[i], width,
                color=train_color, hatch=train_hatches[i],
                edgecolor=edge, linewidth=0.8,
                label="_nolegend_")   # no hatch in legend

        plt.bar(x[i] + width/2, valid_vals[i], width,
                color=valid_color, hatch=valid_hatches[i],
                edgecolor=edge, linewidth=0.8,
                label="_nolegend_")   # no hatch in legend

    plt.ylabel("Accuracy (%)")
    plt.title("Final Training vs Validation Accuracy")
    plt.xticks(x, labels)

    # LEGEND: plain colors only (NO hatch)
    legend_elements = [
        Patch(facecolor=train_color, edgecolor="black", label="Training"),
        Patch(facecolor=valid_color, edgecolor="black", label="Validation"),
    ]
    plt.legend(handles=legend_elements, loc="lower center", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "final_acc_bar.png"), dpi=150)
    plt.savefig(os.path.join(out_dir, "final_acc_bar.pdf"))
    plt.close()


def plot_single_dataset_curves(hist_path, dataset_label, out_dir="./result", max_epochs=30):
    if not os.path.exists(hist_path):
        print("Skipping %s curves: history file missing." % dataset_label)
        return

    train_loss, valid_loss, train_acc, valid_acc = load_histories(hist_path)
    min_len = len(train_loss)
    if max_epochs is not None:
        min_len = min(min_len, int(max_epochs))
    epochs = np.arange(1, min_len + 1)

    train_loss = train_loss[:min_len]
    valid_loss = valid_loss[:min_len]
    train_acc = train_acc[:min_len]
    valid_acc = valid_acc[:min_len]

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(5.0, 3.8), dpi=300)
    plt.plot(epochs, train_loss, label="Training",
           color="blue", linestyle="-", linewidth=2.2, )
    plt.plot(epochs, valid_loss, label="Validation",
           color="red", linestyle="-",  linewidth=2.2)

    plt.xlabel("Epochs", fontsize=13)
    plt.ylabel("Loss", fontsize=13)
    plt.title(f"Loss over Epochs ({dataset_label})", fontsize=13)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.legend(loc="upper right", frameon=True, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_label.lower()}_loss_curve.pdf"))
    plt.savefig(os.path.join(out_dir, f"{dataset_label.lower()}_loss_curve.png"), dpi=300)
    plt.close()


    # ---- ACCURACY ----
    plt.figure(figsize=(7.5, 4.5), dpi=300)
    plt.plot(epochs, np.array(train_acc) * 100.0, label="Training",
           color="green" , linestyle="-",  linewidth=2.2, 
    )
    plt.plot(epochs, np.array(valid_acc) * 100.0, label="Validation",
             color="orange", linestyle="-", linewidth=2.2)

    plt.xlabel("Epochs", fontsize=13)
    plt.ylabel("Accuracy (\%)", fontsize=13)
    plt.title(f"Accuracy over Epochs ({dataset_label})", fontsize=13)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_label.lower()}_acc_curve.pdf"))
    plt.savefig(os.path.join(out_dir, f"{dataset_label.lower()}_acc_curve.png"), dpi=300)
    plt.close()





def plot_metrics_comparison(re_metrics_path, ts_metrics_path, out_dir="./result"):
    if not (os.path.exists(re_metrics_path) and os.path.exists(ts_metrics_path)):
        print("Skipping metrics comparison: metrics files missing.")
        return

    re_metrics = load_metrics(re_metrics_path)
    ts_metrics = load_metrics(ts_metrics_path)

    metric_names = ["accuracy", "precision", "recall", "f1", "auc"]
    re_vals = [re_metrics.get(m, 0.0) * 100.0 for m in metric_names]
    ts_vals = [ts_metrics.get(m, 0.0) * 100.0 for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.15

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7.5, 4.5))

    # ONLY TWO COLORS
    re_color = "#1f77b4"   # blue
    ts_color = "#ff7f0e"   # golden/orange

    # clean edges
    edge = "black"
    plt.rcParams["hatch.linewidth"] = 0.8

    re_hatches = ["///", "xx", "\\\\", "..", "++"]
    ts_hatches = ["\\\\", "oo", "--", "xx", "///"]

    for i in range(len(metric_names)):
        # IMPORTANT: label="_nolegend_" so legend doesn't pick hatched bars
        plt.bar(x[i] - width/2, re_vals[i], width,
                color=re_color, hatch=re_hatches[i],
                edgecolor=edge, linewidth=0.8,
                label="_nolegend_")

        plt.bar(x[i] + width/2, ts_vals[i], width,
                color=ts_color, hatch=ts_hatches[i],
                edgecolor=edge, linewidth=0.8,
                label="_nolegend_")

    plt.ylabel("Score (%)")
    plt.title("Evaluation Metrics Comparison")
    plt.xticks(x, [m.upper() for m in metric_names])

    # LEGEND: only two plain color boxes (NO hatch)
    legend_elements = [
        Patch(facecolor=re_color, edgecolor="black", label="Reentrancy"),
        Patch(facecolor=ts_color, edgecolor="black", label="Timestamp"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_comparison.png"), dpi=150)
    plt.savefig(os.path.join(out_dir, "metrics_comparison.pdf"))
    plt.close()


def plot_all_comparisons(out_dir="./result"):
    re_hist_path = os.path.join(out_dir, "reentrancy_hist.json")
    ts_hist_path = os.path.join(out_dir, "timestamp_hist.json")
    re_metrics_path = os.path.join(out_dir, "reentrancy_metrics.json")
    ts_metrics_path = os.path.join(out_dir, "timestamp_metrics.json")

    plot_accuracy_loss_curves(re_hist_path, ts_hist_path, out_dir=out_dir)
    plot_final_accuracy_bars(re_hist_path, ts_hist_path, out_dir=out_dir)
    plot_metrics_comparison(re_metrics_path, ts_metrics_path, out_dir=out_dir)
    plot_single_dataset_curves(re_hist_path, "Reentrancy", out_dir=out_dir, max_epochs=None)
    plot_single_dataset_curves(ts_hist_path, "Timestamp", out_dir=out_dir, max_epochs=None)

def moving_average(x, w=15):
    x = np.array(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")


def plot_clean_curves(hist_path, dataset_name, out_dir="./result",
                      max_epochs=250, smooth_window=15, loss_log=False):
    if not os.path.exists(hist_path):
        print("Missing:", hist_path)
        return

    train_loss, valid_loss, train_acc, valid_acc = load_histories(hist_path)

    n = min(len(train_loss), len(valid_loss), len(train_acc), len(valid_acc), max_epochs)
    train_loss, valid_loss = train_loss[:n], valid_loss[:n]
    train_acc, valid_acc   = train_acc[:n], valid_acc[:n]
    epochs = np.arange(1, n+1)

    os.makedirs(out_dir, exist_ok=True)

    # ---- ACCURACY ----
    plt.figure(figsize=(5.0, 3.8), dpi=300)
    plt.plot(epochs, np.array(train_acc) * 100.0, label="Training",
            linestyle="-", marker="o", linewidth=2.2, markersize=4,
            markevery=max(1,n //8))
    plt.plot(epochs, np.array(valid_acc) * 100.0, label="Validation",
            linestyle="--", marker="s", linewidth=2.2, markersize=4,
            markevery=max(1, n//8))

    plt.xlabel("Epochs", fontsize=13)
    plt.ylabel("Accuracy (\%)", fontsize=13)
    plt.title(f"Accuracy over Epochs ({dataset_name})", fontsize=13)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name.lower()}_acc_curve.pdf"))
    plt.savefig(os.path.join(out_dir, f"{dataset_name.lower()}_acc_curve.png"), dpi=300)
    plt.close()


    # ---- LOSS ----
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_loss, alpha=0.25, linewidth=1, label="Train (raw)")
    plt.plot(epochs, valid_loss, alpha=0.25, linewidth=1, label="Valid (raw)")

    tl_s = moving_average(train_loss, smooth_window)
    vl_s = moving_average(valid_loss, smooth_window)
    ep_s2 = epochs[len(epochs)-len(tl_s):]

    plt.plot(ep_s2, tl_s, linewidth=2.5, label=f"Train (smooth w={smooth_window})")
    plt.plot(ep_s2, vl_s, linewidth=2.5, label=f"Valid (smooth w={smooth_window})")

    if loss_log:
        plt.yscale("log")

    plt.title(f"Loss over Epochs ({dataset_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linewidth=0.7, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name.lower()}_loss_clean.png"), dpi=200)
    plt.close()

    print("Saved clean curves for", dataset_name)



class DetectModel(object):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 250,
            'patience': 60,
            'learning_rate': 0.002,
            'clamp_gradient_norm': 0.9,  # [0.8, 1.0]
            'out_layer_dropout_keep_prob': 0.8,  # [0.8, 1.0]

            'hidden_size': 256,  # 256/512/1024/2048
            'use_graph': True,

            'tie_fwd_bkwd': False,  # True or False
            'task_ids': [0],
            'propagation_rounds': 2,
            'propagation_substeps': 20,

            'readout_num_heads': 4,
            'readout_attn_hidden': 128,
            'fc_dropout_keep_prob': 0.8,   # 0.7–0.8 try
            'weight_decay': 2e-4,          # L2 regularization

        #   'train_file': 'train_data/reentrancy/train.json',
        #    'valid_file': 'train_data/reentrancy/valid.json'

            'train_file': 'train_data/timestamp/train.json',
            'valid_file': 'train_data/timestamp/valid.json'

      
        }

    def __init__(self, args):
        self.args = args

        # Collect argument things:
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        # random_seed = None
        random_seed = args.get('--random_seed')
        self.random_seed = int(9930)   # optional

        threshold = args.get('--thresholds')
        self.threshold = float(0.45)   # optional

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = args.get('--log_dir') or '.'
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)

        # Collect parameters:
        params = self.default_params()
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))
        self.params = params
        print("debug Starting data loading...")
        print("TRAIN:", os.path.join(self.data_dir, self.params['train_file']))
        print("VALID:", os.path.join(self.data_dir, self.params['valid_file']))

        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        print("Run with current seed %s " % self.random_seed)

        # Load baseline:
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.num_graph = 1
        self.train_num_graph = 0
        self.valid_num_graph = 0
            # Load training and validation data
        self.train_data, self.train_num_graph = self.load_data(params['train_file'], is_training_data=True)
        self.valid_data, self.valid_num_graph = self.load_data(params['valid_file'], is_training_data=False)
            # Build the actual model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()

            # Restore/initialize variables:
            restore_file = args.get('--restore')
            if restore_file is not None:
                self.restore_model(restore_file)
            else:
                self.initialize_model()

    def load_data(self, file_name, is_training_data: bool):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading baseline from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        # Get some common baseline out:
        num_fwd_edge_types = 0
        for g in data:
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        self.annotation_size = max(self.annotation_size, len(data[0]["node_features"][0]))

        return self.process_raw_graphs(data, is_training_data)

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def make_model(self):
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                            name='target_values')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                          name='target_mask')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')

        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [],
                                                                          name='out_layer_dropout_keep_prob')

        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            # This does the actual graph work: (message)
            if self.params['use_graph']:
                self.ops['final_node_representations'] = self.compute_final_node_representations()   #TMP call 
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['process_raw_graphs'])

        self.ops['losses'] = []
        for (internal_id, task_id) in enumerate(self.params['task_ids']):     
            with tf.variable_scope("out_layer_task%i" % task_id):
         #       with tf.variable_scope("regression_gate"):
         #           self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 1, [],
          #                                                                 self.placeholders[
           #                                                                    'out_layer_dropout_keep_prob'])
        #        with tf.variable_scope("regression"):
         #           self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 1, [],
          #                                                                      self.placeholders[
           #                                                                         'out_layer_dropout_keep_prob'])
                # ---- Multi-head attention readout (replaces gated_regression)
                graph_emb = self.multihead_attention_readout(self.ops['final_node_representations'])  # [G, heads*hidden]     #MHA call

                # ---- Simple classifier (FC -> logit)
                with tf.variable_scope("mh_attn_classifier"):      #classifier call
                    graph_emb_drop = tf.layers.dropout(
                        graph_emb,
                        rate=1.0 - self.params['fc_dropout_keep_prob'],
                        training=tf.less(self.placeholders['out_layer_dropout_keep_prob'], 1.0)  

                    )
                    computed_values_2d = tf.layers.dense(
                        graph_emb_drop, 1, activation=None, name="fc_out",
                        kernel_regularizer=tf.keras.regularizers.l2(self.params['weight_decay'])
                    )  # [G,1]
                computed_values = tf.squeeze(computed_values_2d, axis=1)  # [G]
                new_computed_values = tf.nn.sigmoid(computed_values)
                self.ops['logits_task%i' % task_id] = computed_values
                self.ops['prob_task%i' % task_id] = new_computed_values   # sigmoid probs


                # same as before  loss computation
                
                new_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=computed_values,
                        labels=self.placeholders['target_values'][internal_id, :]
                    )
                )

                # optional: keep these for debugging similar to old code
                self.ops['sigm_val'] = graph_emb
                self.ops['initial_re'] = self.placeholders['initial_node_representation']

                #computed_values, sigm_val, initial_re = self.gated_regression(self.ops['final_node_representations'],
                 #                                                             self.weights[
                  #                                                                'regression_gate_task%i' % task_id],
                   #                                                           self.weights[
                    #                                                              'regression_transform_task%i' % task_id])

                def f(x):
                    x = 1 * x
                    x = x.astype(np.float32)
                    return x

               # new_computed_values = tf.nn.sigmoid(computed_values)
               # new_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=computed_values,
                #                                                                  labels=self.placeholders[
                 #                                                                            'target_values'][
                  #                                                                       internal_id, :]))
                a = tf.math.greater_equal(new_computed_values, self.threshold)
                a = tf.py_func(f, [a], tf.float32)
                correct_pred = tf.equal(a, self.placeholders['target_values'][internal_id, :])
                self.ops['new_computed_values'] = new_computed_values
                #self.ops['sigm_val'] = sigm_val  # QP:graph feature
                self.ops['sigm_val'] = graph_emb
                #self.ops['initial_re'] = initial_re  # QP:inital nodes
                self.ops['initial_re'] = self.placeholders['initial_node_representation']  # QP:initial nodes

                self.ops['accuracy_task%i' % task_id] = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                b = tf.multiply(self.placeholders['target_values'][internal_id, :], 2)
                b = tf.py_func(f, [b], tf.float32)
                c = tf.cast(a, tf.float32)
                d = tf.math.add(b, c)
                self.ops['sigm_c'] = correct_pred

                d_TP = tf.math.equal(d, 3)
                TP = tf.reduce_sum(tf.cast(d_TP, tf.float32))  # true positive
                d_FN = tf.math.equal(d, 2)
                FN = tf.reduce_sum(tf.cast(d_FN, tf.float32))  # false negative
                d_FP = tf.math.equal(d, 1)
                FP = tf.reduce_sum(tf.cast(d_FP, tf.float32))  # false positive
                d_TN = tf.math.equal(d, 0)
                TN = tf.reduce_sum(tf.cast(d_TN, tf.float32))  # true negative
                self.ops['sigm_sum'] = tf.add_n([TP, FN, FP, TN])
                self.ops['sigm_TP'] = TP
                self.ops['sigm_FN'] = FN
                self.ops['sigm_FP'] = FP
                self.ops['sigm_TN'] = TN

                R = tf.cast(tf.divide(TP, tf.add(TP, FN)), tf.float32)  # Recall
                P = tf.cast(tf.divide(TP, tf.add(TP, FP)), tf.float32)  # Precision
                FPR = tf.cast(tf.divide(FP, tf.add(TN, FP)), tf.float32)  # FPR: false positive rate
                D_TP = tf.add(TP, TP)
                F1 = tf.cast(tf.divide(D_TP, tf.add_n([D_TP, FP, FN])), tf.float32)  # F1 score
                self.ops['sigm_Recall'] = R
                self.ops['sigm_Precision'] = P
                self.ops['sigm_F1'] = F1
                self.ops['sigm_FPR'] = FPR
                self.ops['losses'].append(new_loss)
        self.ops['loss'] = tf.reduce_sum(self.ops['losses'])
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if reg_losses:
            self.ops['loss'] = self.ops['loss'] + tf.add_n(reg_losses)


    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.get('--freeze-graph-model'):
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        # Initialize newly-introduced variables:
        self.sess.run(tf.local_variables_initializer())

    def gated_regression(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):

        raise Exception("Models have to implement make_minibatch_iterator!")

    def run_epoch(self, epoch_name: str, data, epoch, is_training: bool):
        chemical_accuracies = np.array([0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586,
                                        0.037467458])

        loss = 0
        accuracies = []
        start_time = time.time()
        processed_graphs = 0
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if is_training:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params[
                    'out_layer_dropout_keep_prob']
                fetch_list = [self.ops['loss'], accuracy_ops, self.ops['train_step']]
            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss'], accuracy_ops]
            val_1, val_2, val_3, val_4, val_5, val_6 = self.sess.run(
                [self.ops['sigm_c'], self.ops['sigm_TP'], self.ops['sigm_FN'], self.ops['sigm_FP'], self.ops['sigm_TN'],
                 self.ops['sigm_sum']], feed_dict=batch_data)
            val_R, val_P, val_F1, val_FPR = self.sess.run(
                [self.ops['sigm_Recall'], self.ops['sigm_Precision'], self.ops['sigm_F1'], self.ops['sigm_FPR']],
                feed_dict=batch_data)

            # output the feature vectors (QP)
            if epoch == 150 and is_training is True:
                var_fn = self.sess.run([self.ops['sigm_val']], feed_dict=batch_data)

                ss = tf.unsorted_segment_sum(data=self.ops['final_node_representations'],
                                             segment_ids=self.placeholders['graph_nodes_list'],
                                             num_segments=self.placeholders['num_graphs'])
                var_finial_node = self.sess.run([ss], feed_dict=batch_data)
                np.savetxt("./features/timestamp/timestamp_train_feature.txt", var_finial_node[0],
                           fmt="%.6f")
                # print("graph representation: {}".format(var_fn))
                print("type: {}  length: {}".format(type(var_fn), len(var_fn)))
            elif epoch == 150 and is_training is not True:
                var_fn = self.sess.run([self.ops['sigm_val']], feed_dict=batch_data)
                ss = tf.unsorted_segment_sum(data=self.ops['final_node_representations'],
                                             segment_ids=self.placeholders['graph_nodes_list'],
                                             num_segments=self.placeholders['num_graphs'])
                var_finial_node = self.sess.run([ss], feed_dict=batch_data)
                np.savetxt("./features/timestamp/timestamp_valid_feature.txt", var_finial_node[0],
                           delimiter=", ",
                           fmt="%.6f")
                # print("graph representation: {}".format(var_fn))
                print("type: {}  length: {}".format(type(var_fn), len(var_fn)))

            result = self.sess.run(fetch_list, feed_dict=batch_data)
            if is_training:
                (batch_loss, batch_accuracies) = (result[0], result[1])
            else:
                (batch_loss, batch_accuracies) = (result[0], result[1])
            loss += batch_loss * num_graphs
            accuracies.append(np.array(batch_accuracies) * num_graphs)

            print("random seed: {}".format(self.random_seed))
            print("sum: {}".format(val_6))
            print("TP： {}".format(val_2))
            print("FN： {}".format(val_3))
            print("FP： {}".format(val_4))
            print("TN： {}".format(val_5))
            print("Recall: {}".format(val_R))
            print("Precision: {}".format(val_P))
            print("F1: {}".format(val_F1))
            print("FPR: {}".format(val_FPR))
            print("Running %s, batch %i (has %i graphs). "
                  "Loss so far: %.4f" % (epoch_name, step, num_graphs, loss / processed_graphs), end='\r')

        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        error_ratios = accuracies / chemical_accuracies[self.params["task_ids"]]
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return loss, accuracies, error_ratios, instance_per_sec
    def collect_probs_and_labels(self, data):
        """
        Returns:
          y_true: [N]
          y_prob: [N]  (sigmoid probabilities)
        """
        y_true_all = []
        y_prob_all = []

        task_internal_id = 0  # since task_ids=[0]
        task_id = self.params['task_ids'][0]
        prob_op = self.ops['new_computed_values']

        batch_iterator = ThreadedIterator(
            self.make_minibatch_iterator(data, is_training=False),
            max_queue_size=5
        )

        for batch_data in batch_iterator:
            batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0

            # sigmoid probabilities
            probs = self.sess.run(prob_op, feed_dict=batch_data)
            probs = np.array(probs).reshape(-1)

            # true labels
            y_batch = batch_data[self.placeholders['target_values']][task_internal_id]
            y_batch = np.array(y_batch).reshape(-1)
            # Convert y_batch to integers (0 and 1)
            y_batch = y_batch.astype(int)
            y_prob_all.extend(probs.tolist())
            y_true_all.extend(y_batch.tolist())

        return np.array(y_true_all), np.array(y_prob_all)


    def train(self):
        train_loss_hist, valid_loss_hist = [], []
        train_acc_hist, valid_acc_hist = [], []
        val_acc1 = []
        log_to_save = []
        total_time_start = time.time()
        best_val_loss = float("inf")
        best_val_epoch = 0
        with self.graph.as_default():
            if self.args.get('--restore') is not None:
                _, valid_accs, _, _ = self.run_epoch("Resumed (validation)", self.valid_data, False)
                best_val_acc = np.sum(valid_accs)
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            else:
                (best_val_acc, best_val_acc_epoch) = (float("+inf"), 0)
            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                train_start = time.time()
                self.num_graph = self.train_num_graph
                train_loss, train_accs, train_errs, train_speed = self.run_epoch("epoch %i (training)" % epoch,
                                                                                 self.train_data, epoch, True)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], train_errs)])
                print("\r\x1b[K Train: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (train_loss,
                                                                                                        accs_str,
                                                                                                        errs_str,
                                                                                                        train_speed))
                train_loss_hist.append(float(train_loss))
                train_acc_hist.append(float(np.sum(train_accs)))  # single task => same

                epoch_time_train = time.time() - train_start
                print(epoch_time_train)

                valid_start = time.time()
                self.num_graph = self.valid_num_graph
                valid_loss, valid_accs, valid_errs, valid_speed = self.run_epoch("epoch %i (validation)" % epoch,
                                                                                 self.valid_data, epoch, False)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], valid_errs)])
                print("\r\x1b[K Valid: loss: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (valid_loss,
                                                                                                        accs_str,
                                                                                                        errs_str,
                                                                                                        valid_speed))
                valid_loss_hist.append(float(valid_loss))
                valid_acc_hist.append(float(np.sum(valid_accs)))
                if valid_loss < best_val_loss:
                    print("Validation loss improved from %.5f to %.5f. Saving model."
                         % (best_val_loss, valid_loss))
                    best_val_loss = valid_loss
                    best_val_epoch = epoch
                    self.save_model(self.best_model_file)
                elif epoch - best_val_epoch >= self.params['patience']:
                    print("Early stopping! Validation loss did not improve for %d epochs."
                          % self.params['patience'])
                    print("  Best epoch was:", best_val_epoch)
                    break

                epoch_time_valid = time.time() - valid_start
                print(epoch_time_valid)
                val_acc1.append(valid_accs)

                epoch_time_total = time.time() - total_time_start
                print(epoch_time_total)
                log_entry = {
                    'epoch': epoch,
                    'time': epoch_time_total,
                    'train_results': (train_loss, train_accs.tolist(), train_errs.tolist(), train_speed),
                    'valid_results': (valid_loss, valid_accs.tolist(), valid_errs.tolist(), valid_speed),
                }
                log_to_save.append(log_entry)

            dataset_name = infer_dataset_name(self.params['train_file'])
            histories_path = os.path.join("result", "%s_hist.json" % dataset_name)
            save_histories(
                histories_path,
                train_loss_hist,
                valid_loss_hist,
                train_acc_hist,
                valid_acc_hist
            )
            print("Saved:", histories_path)

            y_true, y_prob = self.collect_probs_and_labels(self.valid_data)
            metrics = compute_eval_metrics(y_true, y_prob, threshold=self.threshold)
            metrics_path = os.path.join("result", "%s_metrics.json" % dataset_name)
            save_metrics(metrics_path, metrics)
            print("Saved:", metrics_path)

            plot_all_comparisons(out_dir="./result")


    def save_model(self, path: str) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
            "params": self.params,
            "weights": weights_to_save
        }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def restore_model(self, path: str) -> None:
        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        # Assert that we got the same model configuration
        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            # Fine to have different task_ids:
            if par not in ['task_ids', 'num_epochs']:
                assert par_value == data_to_load['params'][par]

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)
