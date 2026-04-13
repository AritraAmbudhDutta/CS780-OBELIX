import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_dim=18, output_dim=5):
        """
        Deep Q-Network for the OBELIX robot.
        Input: 18 bits (16 sonar, 1 IR, 1 attachment) 
        Output: 5 actions ("L45", "L22", "FW", "R22", "R45") 
        """
        super(Net, self).__init__()
        
        # Define a hidden layer size (e.g., 64 or 128 as per typical DRL tasks)
        hidden_size = 64 
        
        # Input Layer: Processes the 18-bit observation vector [cite: 121]
        self.fc1 = nn.Linear(input_dim, hidden_size)
        
        # Hidden Layer: Provides additional depth for function approximation
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Output Layer: Predicts Q-values for the 5 discrete actions [cite: 126]
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        """
        Forward pass using ReLU activation for hidden layers.
        """
        # First layer with ReLU
        x = F.relu(self.fc1(x))
        
        # Second layer with ReLU
        x = F.relu(self.fc2(x))
        
        # Output layer returns raw Q-values (logits) for each action
        return self.fc3(x)

def createValueNetwork(input_dim, output_dim):
    """
    Helper function to instantiate the network.
    Called by the DDQN class during initialization.
    """
    return Net(input_dim, output_dim)