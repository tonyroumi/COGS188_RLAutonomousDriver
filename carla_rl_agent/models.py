import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNet(nn.Module):
    """
    A convolutional neural network for both policy (actor) and value (critic) estimation.
    
    Parameters:
        num_actions: Number of possible discrete actions.
        
    Returns:
        policy_logits: Raw scores for each action.
        value: Estimated state value.
    """
    def __init__(self, num_actions):
        super(ActorCriticNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_common = nn.Linear(32 * 15 * 23, 128)
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

        # Xavier initialization for better balanced actor outputs
        nn.init.xavier_uniform_(self.actor.weight)
        self.actor.bias.data.fill_(0.0)

    def forward(self, x):
        """
        Pass input through the network.
        
        Parameters:
            x: Input tensor (image).
            
        Returns:
            policy_logits: Logits for action probabilities.
            value: Estimated state value.
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_common(x))
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value 