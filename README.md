# Deep Q-Learning - Lunar Lander

## Assignment Information

- **Student Name:** Mpayimana Cyiza Landry
- **Assignment Title:** Deep Q-Learning - Lunar Lander
- **School:** Stanford University
- **Course:** Reinforcement Learning

---

## Assignment Overview

In this assignment, I implemented a Deep Q-Learning (DQN) agent to solve the Lunar Lander environment from OpenAI Gym. The goal was to train an agent to safely land a lunar lander on a landing pad on the moon's surface. This involved implementing a Deep Q-Network with experience replay and target networks to achieve stable learning.

---

## What I Learned

### 1. **Reinforcement Learning Fundamentals**
- Understood the agent-environment interaction loop
- Learned about states, actions, rewards, and episodes
- Understood the concept of Markov Decision Processes (MDPs)

### 2. **Deep Q-Learning**
- Implemented Q-learning with neural networks (DQN)
- Learned the Bellman equation for action-value functions
- Understood how to approximate Q(s,a) using deep neural networks

### 3. **Stabilization Techniques**
- **Target Network**: Implemented a separate network for generating stable targets
- **Experience Replay**: Used memory buffer to break correlations in sequential data
- **Soft Updates**: Implemented gradual weight updates for target network

### 4. **Lunar Lander Environment**
- Understood the action space (4 discrete actions)
- Learned about observation space (8 continuous state variables)
- Understood reward structure and episode termination conditions

### 5. **TensorFlow Implementation**
- Used Keras Sequential API for building neural networks
- Implemented custom training loops with GradientTape
- Used @tf.function decorator for performance optimization

---

## What I Accomplished

### ✅ Implemented Core Functions

1. **Q-Network and Target Network**
   - Built neural network architecture with:
     - Input layer (state_size)
     - 2 hidden layers (64 units each, ReLU activation)
     - Output layer (num_actions, linear activation)

2. **`compute_loss(experiences, gamma, q_network, target_q_network)`**
   - Implemented the loss calculation for DQN
   - Set target values using Bellman equation
   - Handled terminal states appropriately
   - Computed Mean-Squared Error (MSE)

3. **`agent_learn(experiences, gamma)`**
   - Implemented gradient descent using GradientTape
   - Updated Q-network weights
   - Performed soft update of target network

### ✅ Trained a Successful Agent

- **Training Episodes**: 2000 (solved in ~500-800 episodes)
- **Time Steps**: Maximum 1000 per episode
- **Success Criteria**: Average score ≥ 200 over 100 episodes
- **Agent Performance**: Consistently lands safely on landing pad
- **Training Runtime**: 10-15 minutes

### ✅ Achieved Understanding

- Discovered the importance of experience replay in breaking correlations
- Learned how target networks stabilize training
- Understood the role of hyperparameters (GAMMA, ALPHA, epsilon decay)
- Visualized training progress through reward plots

---

## Environment Details

### Lunar Lander Environment

| Component | Description |
|-----------|-------------|
| **Goal** | Land safely on landing pad at (0,0) |
| **Actions** | 4 discrete: Do nothing, Fire right, Fire main, Fire left |
| **State Space** | 8 variables (position, velocity, angle, angular velocity, leg contact) |
| **Reward** | Based on distance, speed, angle, engine usage, leg contact |
| **Episode Termination** | Crash, exceed x-boundary, or 1000 steps |

### Action Space

| Action | Value | Description |
|--------|-------|-------------|
| Do nothing | 0 | No engine firing |
| Fire right engine | 1 | Fire right thruster |
| Fire main engine | 2 | Fire main thruster (downward) |
| Fire left engine | 3 | Fire left thruster |

### Observation Space

State vector (8 variables):
1. **x** - Horizontal position
2. **y** - Vertical position
3. **vx** - Horizontal velocity
4. **vy** - Vertical velocity
5. **θ** - Angle (radians)
6. **ω** - Angular velocity
7. **l** - Left leg contact (boolean)
8. **r** - Right leg contact (boolean)

---

## Technical Implementation

### Technologies Used
- **Python 3**
- **TensorFlow** - Deep learning framework
- **Keras** - Neural network API
- **OpenAI Gym** - Environment toolkit
- **NumPy** - Numerical computations
- **PIL/ImageIO** - Video rendering

### Neural Network Architecture

```python
q_network = Sequential([
    Input(shape=state_size),  # (8,)
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_actions, activation='linear')  # (4,)
])