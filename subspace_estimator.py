import torch.nn as nn
import torch

class SubspaceEstimator(nn.Module):
    def __init__(self, embedding_dimension=128, num_basis_functions=64, subspace_rank = 4):
        super(SubspaceEstimator, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.subspace_rank = subspace_rank
        
        self.coeff_net = nn.Sequential(
            nn.Linear(embedding_dimension, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

        
        self.mu_basis = nn.Parameter(torch.randn(self.num_basis_functions))
        self.alpha_basis = nn.Parameter(torch.randn(self.num_basis_functions))
        self.beta_basis = nn.Parameter(torch.randn(self.num_basis_functions))
        self.gamma_basis = nn.Parameter(torch.randn(self.num_basis_functions))
        
        self.activation = nn.ReLU()
    
    def calculate_basis_functions(self, t):
        
        t_reshaped = t.unsqueeze(-1) # (Batch, SequenceLength, 1)
        
        mu_reshaped = self.mu_basis.view(1,1,-1) # (1,1,N=64)
        alpha_reshaped = self.alpha_basis.view(1,1,-1) # (1,1,N=64)
        beta_reshaped = self.beta_basis.view(1,1,-1) # (1,1,N=64)
        gamma_reshaped = self.gamma_basis.view(1,1,-1) # (1,1,N=64)
        
        term1 = torch.exp(-(alpha_reshaped*(t_reshaped-mu_reshaped))**2)
        term2 = torch.cos(beta_reshaped*t_reshaped + gamma_reshaped)
        h_t = term1 * term2
        return h_t # (Batch, SequenceLength, N=64)
    
    def calculate_H_matrix(self, h_t):
        
        batch_size, seq_len, N = h_t.shape
        H_width = 2*N
        H_height = 2*seq_len
        
        H = torch.zeros(batch_size, H_height, H_width, device=h_t.device)
        
        H[:, 0::2, 0:N] = h_t
        H[:, 1::2, N:H_width] = h_t
        return H
        
    def forward(self, f, t):
        batch_size = f.shape[0]
        
        h_t = self.calculate_basis_functions(t)
        H = self.calculate_H_matrix(h_t)
        
        coefficients = self.coeff_net(f)
        omega = coefficients.view(batch_size, 2 * self.num_basis_functions, self.subspace_rank)
        B = torch.bmm(H, omega)
        return B