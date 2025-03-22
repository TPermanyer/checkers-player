import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from environment import CheckersEnv  # Importa tu entorno
import matplotlib as plt


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


# Define the Dueling Q-Network using CNN.
class DuelingQNetwork(nn.Module):
    def __init__(self, output_dim):
        super(DuelingQNetwork, self).__init__()
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Dueling architecture
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # flatten
        
        advantage = self.advantage_stream(x)
        value = self.value_stream(x)
        
        # Combine value and advantage
        return value + advantage - advantage.mean(dim=1, keepdim=True)

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

class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6, beta_start=0.4):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use
        self.beta = beta_start  # Importance sampling correction
        self.beta_increment = 0.0001
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) < self.capacity:
            prob_weights = self.priorities[:len(self.buffer)]
        else:
            prob_weights = self.priorities
        
        prob_weights = prob_weights ** self.alpha
        prob_weights = prob_weights / prob_weights.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=prob_weights)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * prob_weights[indices]) ** -self.beta
        weights = weights / weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Small constant to ensure non-zero probability
    
    def __len__(self):
        return len(self.buffer)

def train_dqn_improved(env, num_episodes=10, batch_size=128, gamma=0.99, lr=3e-4,
                     epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999954):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use Dueling network
    output_dim = 256  # Based on your move index mapping
    q_net = DuelingQNetwork(output_dim).to(device)
    target_net = DuelingQNetwork(output_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    
    # Use Adam optimizer with learning rate scheduler
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50)
    
    # Prioritized replay buffer
    replay_buffer = PrioritizedReplayBuffer()
    epsilon = epsilon_start
    
    # Metrics tracking
    rewards_history = []
    avg_rewards = []
    loss_history = []
    best_avg_reward = -float('inf')
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_losses = []
        
        while not done:
            # Get board state and legal moves
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            legal_moves, _ = env.get_legal_moves(env.current_player)
            
            
                
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(legal_moves)
            else:
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                    # Filter for legal moves only
                    legal_actions = {move: q_values[0, move_index(move)].item() for move in legal_moves}
                    action = max(legal_actions.items(), key=lambda x: x[1])[0]
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Store in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            
            # Training step
            if len(replay_buffer) >= batch_size:
                # Sample with priorities
                (states, actions, rewards, next_states, dones), indices, weights = replay_buffer.sample(batch_size)
                
                # Convert to tensors
                states_tensor = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(1).to(device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).unsqueeze(1).to(device)
                dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
                weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
                
                # Get current Q values
                q_values = q_net(states_tensor)
                action_indices = torch.tensor([move_index(a) for a in actions], dtype=torch.long).to(device)
                current_q = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
                
                # Double DQN target: use online network to select actions, target network to evaluate
                with torch.no_grad():
                    # Select best actions using online network
                    online_q_next = q_net(next_states_tensor)
                    best_actions = online_q_next.max(1)[1]
                    
                    # Evaluate using target network
                    target_q_next = target_net(next_states_tensor)
                    next_q = target_q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards_tensor + gamma * next_q * (1 - dones_tensor)
                
                # Compute TD errors for updating priorities
                td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
                replay_buffer.update_priorities(indices, td_errors)
                
                # Compute loss with importance sampling weights
                loss = (weights_tensor * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()
                
                episode_losses.append(loss.item())
        
        # End of episode processing
        rewards_history.append(total_reward)
        if episode_losses:
            loss_history.append(sum(episode_losses) / len(episode_losses))
        
        # Calculate running average of last 100 episodes
        if len(rewards_history) >= 100:
            avg_reward = sum(rewards_history[-100:]) / 100
            avg_rewards.append(avg_reward)
            
            # Learning rate scheduling based on performance
            scheduler.step(avg_reward)
            
            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(q_net.state_dict(), "best_checker_model.pth")
        else:
            avg_rewards.append(0)
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Soft update target network
        if episode % 5 == 0:
            for target_param, param in zip(target_net.parameters(), q_net.parameters()):
                target_param.data.copy_(0.05 * param.data + 0.95 * target_param.data)
        
        # Regular checkpoints
        if episode % 100 == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': q_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
            }, f"checker_checkpoint_{episode}.pth")
        
        # Logging
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, Reward: {total_reward:.1f}, "
                  f"Avg100: {avg_rewards[-1]:.1f}, Epsilon: {epsilon:.4f}")
    
    # Save final model
    torch.save(q_net.state_dict(), "final_checker_modelS.pth")
    print(f"Training complete. Best average reward: {best_avg_reward:.2f}") 
    
    # Plot training progress
        
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(avg_rewards)
    plt.title('Average Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.subplot(1, 3, 3)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
   
    
    return q_net

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a Checkers DQN agent')
    parser.add_argument('--episodes', type=int, default=20, help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--self-play', action='store_true', help='Use self-play training')
    parser.add_argument('--load', type=str, help='Load model from checkpoint')
    
    args = parser.parse_args()
    
    env = CheckersEnv()
    
    if args.self_play:
        model = train_self_play(
            num_iterations=5,
            games_per_iteration=args.episodes // 5,
            batch_size=args.batch_size,
            lr=args.lr,
            gamma=args.gamma
        )
    else:
        model = train_dqn_improved(
            env,
            num_episodes=args.episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            gamma=args.gamma
        )
