import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import utils
from sklearn import cluster
import pickle
import scipy.sparse as sparse
import time
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from metrics.accuracy import clustering_accuracy
import argparse
import random
from tqdm import tqdm
import os
import csv
from torch.utils.data import DataLoader
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models.trajectory_embedder import TrajectoryEmbeddingModel
from datasets import Hopkins155
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MLP(nn.Module):
    
    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=False):
        super(MLP, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.output_dims = out_dims
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(self.input_dims, self.hid_dims[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hid_dims) - 1):
            self.layers.append(nn.Linear(self.hid_dims[i], self.hid_dims[i + 1]))
            self.layers.append(nn.ReLU())

        self.out_layer = nn.Linear(self.hid_dims[-1], self.output_dims)
        if kaiming_init:
            self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)
        init.xavier_uniform_(self.out_layer.weight)
        init.zeros_(self.out_layer.bias)
        
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        h = self.out_layer(h)
        h = torch.tanh_(h)
        return h


class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))
    
    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)


class SENet(nn.Module):

    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=True):
        super(SENet, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.kaiming_init = kaiming_init
        self.shrink = 1.0 / out_dims

        self.net_q = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.net_k = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.thres = AdaptiveSoftThreshold(1)

    def query_embedding(self, queries):
        q_emb = self.net_q(queries)
        return q_emb
    
    def key_embedding(self, keys):
        k_emb = self.net_k(keys)
        return k_emb

    def get_coeff(self, q_emb, k_emb):
        c = self.thres(q_emb.mm(k_emb.t()))
        return self.shrink * c

    def forward(self, queries, keys):
        q = self.query_embedding(queries)
        k = self.key_embedding(keys)
        out = self.get_coeff(q_emb=q, k_emb=k)
        return out


def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()


def get_sparse_rep(senet, data, batch_size=10, chunk_size=100, non_zeros=1000):
    N, D = data.shape
    non_zeros = min(N, non_zeros)
    C = torch.empty([batch_size, N])
    if (N % batch_size != 0):
        raise Exception("batch_size should be a factor of dataset size.")
    if (N % chunk_size != 0):
        raise Exception("chunk_size should be a factor of dataset size.")

    val = []
    indicies = []
    with torch.no_grad():
        senet.eval()
        for i in range(data.shape[0] // batch_size):
            chunk = data[i * batch_size:(i + 1) * batch_size].cuda()
            q = senet.query_embedding(chunk)
            for j in range(data.shape[0] // chunk_size):
                chunk_samples = data[j * chunk_size: (j + 1) * chunk_size].cuda()
                k = senet.key_embedding(chunk_samples)   
                temp = senet.get_coeff(q, k)
                C[:, j * chunk_size:(j + 1) * chunk_size] = temp.cpu()

            rows = list(range(batch_size))
            cols = [j + i * batch_size for j in rows]
            C[rows, cols] = 0.0

            _, index = torch.topk(torch.abs(C), dim=1, k=non_zeros)
            
            val.append(C.gather(1, index).reshape([-1]).cpu().data.numpy())
            index = index.reshape([-1]).cpu().data.numpy()
            indicies.append(index)
    
    val = np.concatenate(val, axis=0)
    indicies = np.concatenate(indicies, axis=0)
    indptr = [non_zeros * i for i in range(N + 1)]
    
    C_sparse = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])
    return C_sparse


def get_knn_Aff(C_sparse_normalized, k=3, mode='symmetric'):
    C_knn = kneighbors_graph(C_sparse_normalized, k, mode='connectivity', include_self=False, n_jobs=10)
    if mode == 'symmetric':
        Aff_knn = 0.5 * (C_knn + C_knn.T)
    elif mode == 'reciprocal':
        Aff_knn = C_knn.multiply(C_knn.T)
    else:
        raise Exception("Mode must be 'symmetric' or 'reciprocal'")
    return Aff_knn


def evaluate(senet, data, labels, num_subspaces, spectral_dim, non_zeros=1000, n_neighbors=3,
             batch_size=10000, chunk_size=10000, affinity='nearest_neighbor', knn_mode='symmetric'):
    C_sparse = get_sparse_rep(senet=senet, data=data, batch_size=batch_size,
                              chunk_size=chunk_size, non_zeros=non_zeros)
    C_sparse_normalized = normalize(C_sparse).astype(np.float32)
    if affinity == 'symmetric':
        Aff = 0.5 * (np.abs(C_sparse_normalized) + np.abs(C_sparse_normalized).T)
    elif affinity == 'nearest_neighbor':
        Aff = get_knn_Aff(C_sparse_normalized, k=n_neighbors, mode=knn_mode)
    else:
        raise Exception("affinity should be 'symmetric' or 'nearest_neighbor'")
    preds = utils.spectral_clustering(Aff, num_subspaces, spectral_dim)
    acc = clustering_accuracy(labels, preds)
    nmi = normalized_mutual_info_score(labels, preds, average_method='geometric')
    ari = adjusted_rand_score(labels, preds)
    return acc, nmi, ari


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=20.0)
    parser.add_argument('--embedding_model_path', type=str, default='out/models/trained_model_weights_normalized.pt')
    parser.add_argument('--lmbd', type=float, default=0.9)
    parser.add_argument('--hid_dims', type=int, default=[512, 512])
    parser.add_argument('--out_dims', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_min', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--non_zeros', type=int, default=100)
    parser.add_argument('--n_neighbors', type=int, default=3)
    parser.add_argument('--spectral_dim', type=int, default=15)
    parser.add_argument('--affinity', type=str, default="nearest_neighbor")
    parser.add_argument('--mean_subtract', dest='mean_subtraction', action='store_true')
    parser.set_defaults(mean_subtraction=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    same_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embedding_model = TrajectoryEmbeddingModel()

    state_dict = torch.load(args.embedding_model_path, map_location='cpu')
    embedding_model.load_state_dict(state_dict, strict=False)
    feature_extractor = embedding_model.feature_extractor
    feature_extractor.to(device)
    embedding_model.to(device)
    feature_extractor.eval()
    embedding_model.eval()
    print("Feature extractor loaded.")

    hopkins_dataset = Hopkins155(root_dir='data/Hopkins155/')
    hopkins_loader = DataLoader(hopkins_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_accs, all_nmis, all_aris = [], [], []
    results_summary = []
    
    for seq_data_batch in tqdm(hopkins_loader, desc="Processing Sequences"):
        seq_traj = seq_data_batch['trajectories'].squeeze(0)
        seq_labels = seq_data_batch['labels'].squeeze(0).numpy()
        seq_t = seq_data_batch['times'].squeeze(0)
        seq_k = seq_data_batch['num_clusters'].item()
        seq_name = seq_data_batch['name'][0]
      
        num_points, num_frames, _ = seq_traj.shape 

        with torch.no_grad():
            seq_traj_dev = seq_traj.to(device)
            seq_t_dev = seq_t.to(device)
            x_permuted = seq_traj_dev.permute(0, 2, 1)
            f, B = embedding_model(seq_traj_dev, seq_t_dev)
            
            B_flat = B.view(num_points, -1) # (P, 2F*rank)
            v = torch.cat((f, B_flat), dim=1)
            # features_f = feature_extractor(x_permuted).detach().cpu()

        data = F.normalize(v, p=2, dim=1).detach().cpu()
        data = data.float()
        input_dim = data.shape[1]
        
        
        print(f"Training SENet for {seq_name}")
        senet = SENet(input_dim, args.hid_dims, args.out_dims, kaiming_init=True).to(device)
        optimizer = optim.Adam(senet.parameters(), lr=args.lr)

        block_size = num_points
        n_step_per_iter = max(1, round(num_points / block_size))       

        for epoch in range(args.epochs):
            senet.train()
            epoch_rec_loss = 0
            epoch_reg = 0
        
            batch = data.to(device)
            q = senet.query_embedding(batch)
            k = senet.key_embedding(batch)
            
            rec_batch = torch.zeros_like(batch).to(device)
            reg = torch.zeros([1]).to(device)
            for j in range(n_step_per_iter):
                start_idx = j * num_points
                end_idx = min((j + 1) * num_points, num_points)
                if start_idx >= end_idx: continue

                block = batch[start_idx:end_idx]
                k_block = senet.key_embedding(block)
                c = senet.get_coeff(q, k_block)
                rec_batch = rec_batch + c.mm(block)
                reg = reg + regularizer(c, args.lmbd)
            
            k = senet.key_embedding(batch)
            diag_c = senet.thres((q * k).sum(dim=1, keepdim=True)) * senet.shrink
            rec_batch = rec_batch - diag_c * batch
            reg = reg - regularizer(diag_c, args.lmbd)
            
            rec_loss = torch.sum(torch.pow(batch - rec_batch, 2))
            loss = (0.5 * args.gamma * rec_loss + reg) / num_points

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(senet.parameters(), 0.001)
            optimizer.step()
            
            epoch_rec_loss += rec_loss.item()
            epoch_reg += reg.item()

            # if epoch % 20 == 0:
            #   print(f"Epoch {epoch+1}/{args.epochs}, Rec Loss: {epoch_rec_loss/num_points:.4f}, Reg: {epoch_reg/num_points:.4f}")

        print(f"Evaluating SENet for {seq_name}...")
        acc, nmi, ari = evaluate(senet, data=data.cpu(), labels=seq_labels, num_subspaces=seq_k,
                                 affinity=args.affinity, spectral_dim=min(args.spectral_dim, num_points-1),
                                 non_zeros=min(args.non_zeros, num_points-1),
                                 n_neighbors=args.n_neighbors,
                                 batch_size=num_points,
                                 chunk_size=num_points,
                                 knn_mode='symmetric')

        print(f"Results for {seq_name}: ACC-{acc:.4f}, NMI-{nmi:.4f}, ARI-{ari:.4f}")
        results_summary.append({'name': seq_name, 'acc': acc, 'nmi': nmi, 'ari': ari})
        all_accs.append(acc)
        all_nmis.append(nmi)
        all_aris.append(ari)


        torch.cuda.empty_cache()
    mean_acc = np.mean(all_accs)
    mean_nmi = np.mean(all_nmis)
    mean_ari = np.mean(all_aris)
    print("\n--- Overall SENet Performance ---")
    print(f"Mean ACC: {mean_acc:.4f}")
    print(f"Mean NMI: {mean_nmi:.4f}")
    print(f"Mean ARI: {mean_ari:.4f}")
