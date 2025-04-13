import torch.nn as nn

class SubspaceEstimator(nn.Module):
    def __init__(self, embedding_dimension=128, num_basis_functions=64):
        super(SubspaceEstimator, self).__init__()
        
        self.linear1 = nn.Linear(embedding_dimension, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 512)
        
        self.mu_basis = nn.Embedding(num_basis_functions)
        self.alpha_basis = nn.Embedding(num_basis_functions)
        self.beta_basis = nn.Embedding(num_basis_functions)
        self.gamma_basis = nn.Embedding(num_basis_functions)
        
        self.activation = nn.ReLU()
    
    def calculate_basis_functions(self, t):
        h_t_e =
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = x.reshape(128,4)
        return x