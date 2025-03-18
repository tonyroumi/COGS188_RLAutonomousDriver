# CARLA Reinforcement Learning Agent

A simple self-driving agent for the CARLA simulator using reinforcement learning with an Actor-Critic (A3C) architecture.

## Overview

This project implements a reinforcement learning agent that learns to drive in the CARLA simulator environment. The agent uses:

- Actor-Critic (A3C) architecture for policy and value estimation
- Prioritized Experience Replay for efficient learning
- CNN-based perception from camera images
- A* search for global path planning
- Reward shaping for driving behavior

# Setup
**Clone the repoistory and build the environment**
```
git clone git@github.com:tonyroumi/COGS188_RLAutonomousDriver.git carla-rl-agent
cd carla-rl-agent
conda create -n carla-rl-agent python=3.7
conda activate carla-rl-agent
pip3 install -r requirements.txt
```

## Installation

1. Download and setup CARLA 0.10.0
```
chmod +x setup_carla.sh
./setup_carla.sh
```

**Set environment variables**
```
export CARLA_ROOT=/path/to/your/carla/installation
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
```

## Usage

### Starting CARLA

Start the CARLA simulator with:

```
./CarlaUE4.sh -carla-rpc-port=2000 -quality-level=Low -benchmark -fps=30
```

### Training the Agent

```
python scripts/train.py --episodes 1000 --save-dir models/
```

### Evaluating the Agent

```
python scripts/evaluate.py --model-path models/actor_critic_weights.pth
```

