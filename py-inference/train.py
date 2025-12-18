import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys

# Add generated code to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../gen/python'))
import snake_pb2
from model import SnakeNet

DATA_DIR = "data"
MODEL_PATH = "models/latest.pt"
BATCH_SIZE = 64
EPOCHS = 1
LR = 0.001

class SnakeDataset(Dataset):
    def __init__(self, file_paths):
        self.examples = []
        for path in file_paths:
            try:
                with open(path, "rb") as f:
                    data = snake_pb2.TrainingData()
                    data.ParseFromString(f.read())
                    self.examples.extend(data.examples)
            except Exception as e:
                print(f"Error reading {path}: {e}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # State
        state = np.frombuffer(ex.state_data, dtype=np.float32).reshape(17, 11, 11).copy()
        
        # Policy Target
        policy = np.array(ex.policies, dtype=np.float32)
        
        # Value Target
        value = np.array(ex.values, dtype=np.float32)
        
        return torch.from_numpy(state), torch.from_numpy(policy), torch.from_numpy(value)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load Data
    files = glob.glob(os.path.join(DATA_DIR, "*.pb"))
    if not files:
        print("No training data found.")
        return

    print(f"Found {len(files)} data files.")
    dataset = SnakeDataset(files)
    print(f"Loaded {len(dataset)} examples.")
    
    if len(dataset) == 0:
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load Model
    model = SnakeNet(in_channels=17, width=11, height=11).to(device)
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Loss Functions
    # Policy: Cross Entropy (or KL Divergence)
    # Value: MSE
    mse_loss = nn.MSELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for states, policies, values in dataloader:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)
            
            optimizer.zero_grad()
            
            pred_policies, pred_values = model(states)
            
            # Policy Loss: -sum(target * log(pred))
            # Add epsilon to avoid log(0)
            loss_policy = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / states.size(0)
            
            # Value Loss
            loss_value = mse_loss(pred_values, values)
            
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

    # Save Model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Cleanup Data?
    # For now, keep it. In a real loop, we might archive it.

if __name__ == "__main__":
    train()
