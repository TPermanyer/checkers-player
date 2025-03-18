import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from environment import CheckersEnv  # Import your custom environment

# Helper functions for move mapping.
def dark_square_index(row, col):
    """
    Computes a unique index for a dark square.
    In checkers, pieces are placed only on dark squares.
    For even rows, dark squares are at columns 1,3,5,7;
    for odd rows, they are at columns 0,2,4,6.
    """
    if row % 2 == 0:
        return row * 4 + ((col - 1) // 2)
    else:
        return row * 4 + (col // 2)

delta_to_offset = {
    (-1, -1): 0,
    (-1,  1): 1,
    ( 1, -1): 2,
    ( 1,  1): 3,
    (-2, -2): 4,
    (-2,  2): 5,
    ( 2, -2): 6,
    ( 2,  2): 7
}


def move_index(move):
    """
    Maps a move (from_row, from_col, to_row, to_col) to a unique index between 0 and 255.
    """
    fr, fc, tr, tc = move
    ds_index = dark_square_index(fr, fc)
    dr = tr - fr
    dc = tc - fc
    offset = delta_to_offset.get((dr, dc))
    if offset is None:
        raise ValueError(f"Unexpected move delta: {(dr, dc)}")
    return ds_index * 8 + offset

# Define the Q-Network.
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = x.float()
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer.
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def train_dqn(env, num_episodes=500, batch_size=64, gamma=0.99, lr=1e-3,
              epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    # Update output_dim to 256 to match our move_index mapping.
    output_dim = 256
    
    q_net = QNetwork(input_dim, output_dim).to(device)
    target_net = QNetwork(input_dim, output_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            legal_moves = env.get_legal_moves(env.current_player)
            if not legal_moves:
                break

            if random.random() < epsilon:
                action = random.choice(legal_moves)
            else:
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                    # Select the legal move with the highest Q-value using move_index.
                    action = max(legal_moves, key=lambda move: q_values[0, move_index(move)].item())
            
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
                dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
                
                q_vals = q_net(states_tensor)
                # Map each action in the batch to its corresponding index.
                action_indices = torch.tensor([move_index(a) for a in actions], dtype=torch.long).to(device)
                current_q = q_vals.gather(1, action_indices.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    next_q_vals = target_net(next_states_tensor)
                    max_next_q = next_q_vals.max(1)[0]
                    target_q = rewards_tensor + gamma * max_next_q * (1 - dones_tensor)
                
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if episode % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())
        print(f"Episode {episode}, Total Reward: {total_reward}")

    # Save the trained model's state dictionary.
    torch.save(q_net.state_dict(), "trained_q_net_state.pth")
    print("Training complete. Model saved as 'trained_q_net_state.pth'.")

if __name__ == "__main__":
    env = CheckersEnv()
    train_dqn(env)
