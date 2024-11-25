# The Three Climate Experiment

## Overview

This codebase implements a reinforcement learning (RL) framework for optimizing HVAC (Heating, Ventilation, and Air Conditioning) energy consumption in buildings using EnergyPlus simulations. The framework utilizes a Deep Double Q-Network (DDQN) approach to train an agent that learns to make energy-efficient decisions based on environmental data. It has been extended to include a replay across experiments framework shown to make the agent more robust to climate variation.

## Directory Structure

```
three-climate-experiment/
├── OpenStudio-1.4.0/          # EnergyPlus installation directory
├── scripts/                    # Contains executable scripts
│   └── run_experiment.py       # Main script to run the experiment
├── src/                        # Source code for the project
│   ├── __init__.py            # Marks the directory as a package
│   ├── callbacks.py            # Callback functions for EnergyPlus
│   ├── dqn.py                  # DQN agent implementation
│   ├── replay_buffer.py         # Replay buffer for storing experiences
│   └── trainer.py              # Training logic for the DQN agent
├── parameters.txt              # Configuration parameters for the experiment
└── README.md                   # This README file
```

## Requirements

- Python 3.10 or higher
- PyTorch
- Pandas
- NumPy
- EnergyPlus (OpenStudio)
- Additional dependencies listed in `requirements.txt`

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/itsnemoooo/three-climate-experiment
   cd three-climate-experiment
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv NewVenv
   source NewVenv/bin/activate  # On Windows use: NewVenv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up EnergyPlus**:
   - Download and install EnergyPlus from the [EnergyPlus website](https://energyplus.net/).
   - Ensure the path to EnergyPlus is correctly set in the code (see `run_experiment.py`).

## Running the Experiment

To run the experiment, execute the following command:

```bash
python scripts/run_experiment.py
```

### Parameters

The experiment parameters are defined in `parameters.txt`. Key parameters include:

- `state_dim`: Dimension of the state space.
- `action_dim`: Number of possible actions (HVAC actions).
- `lr`: Learning rate for the DQN.
- `gamma`: Discount factor for future rewards.
- `epsilon`: Exploration rate for the agent.
- `epochs`: Number of training epochs.
- `buffer_size`: Size of the replay buffer.
- `RL_flag`: Flag to enable or disable reinforcement learning.

### Weather and Data Files

- **Weather Files**: Located in `data/weather-data/`, these files provide environmental data for simulations.
- **Replay Buffer Files**: Located in `data/buffer/`, these CSV files store experiences for training the DQN agent.

## Code Structure

- **`src/trainer.py`**: Contains the `DQNTrainer` class, which manages the training process, including setting up the baseline and training epochs.
- **`src/callbacks.py`**: Implements callback functions that interact with EnergyPlus during simulation.
- **`src/dqn.py`**: Defines the DQN agent, including methods for action selection and learning from experiences.
- **`src/replay_buffer.py`**: Manages the storage and retrieval of experiences for training the agent.

## Thesis

For a summary of the research and methodologies used in this project, please refer to my thesis executive summary:

[Download Thesis PDF](ExecutiveSummary.pdf)


## Debugging Tips

- If you encounter a `ModuleNotFoundError`, ensure that the `src` directory is included in the Python path.
- For `KeyError` issues, check the state dictionary being passed to the agent and ensure all expected keys are present.