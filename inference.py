import torch
from models.trajectory_embedder import TrajectoryEmbeddingModel


def load_model():    
    model = TrajectoryEmbeddingModel()
    load_path = 'out/models/trained_model_weights.pt'

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(load_path, map_location=target_device)
    
    model.load_state_dict(state_dict, strict=True)
    print("Model weights loaded successfully.")
    model.to(target_device)
    model.eval()
    return model

def main():
    model = load_model()

if __name__ == '__main__':
    main()