import numpy as np


class PrioritizedReplayBuffer:
    """
    Stores transitions and samples them based on a priority metric.
    
    Parameters:
        capacity: Maximum number of transitions to store.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0
    
    def add(self, transition, priority):
        """
        Add a transition with its priority to the buffer.
        
        Parameters:
            transition: Tuple (state, action, reward, return).
            priority: Numeric priority for sampling.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, alpha=0.6):
        """
        Sample a batch of transitions using prioritized experience replay.
        
        Parameters:
            batch_size: Number of transitions to sample.
            alpha: Degree of prioritization.
            
        Returns:
            samples: List of transitions.
            indices: Their indices.
            probs: Sampling probabilities.
        """
        if len(self.buffer) == 0:
            return [], [], []
        
        priorities = np.array(self.priorities) ** alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs)
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices, probs[indices]
    
    def update_priorities(self, indices, errors, epsilon=1e-6):
        """
        Update the priorities for a set of transitions.
        
        Parameters:
            indices: List of indices.
            errors: TD errors for each transition.
            epsilon: Small constant to avoid zero priority.
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + epsilon
    
    def __len__(self):
        return len(self.buffer)