import time
import torch
import numpy as np
from environment import CheckersEnv
from train import DuelingQNetwork, move_index
import random
import os
import matplotlib.pyplot as plt

def evaluate_model(env, model1, model2, num_episodes=100, exploration_rate=0.05):
    """
    Evaluates two models against each other in the checkers environment.
    
    Args:
        env: The checkers environment
        model1: Model for player 1 (red)
        model2: Model for player -1 (white)
        num_episodes: Number of games to play
        exploration_rate: Chance of making a random move
    """
    total_rewards_p1 = []
    total_rewards_p2 = []
    total_moves = []
    wins_p1 = 0
    wins_p2 = 0
    draws = 0
    
    device = next(model1.parameters()).device
    start_time = time.time()
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward_p1 = 0
        ep_reward_p2 = 0
        move_count = 0
        game_winner = None  # Track the winner for this game
        
        while not done and move_count < 200:  # Move limit to prevent infinite games
            legal_moves, has_captures = env.get_legal_moves(env.current_player)
            if not legal_moves:
                # Game over - current player has no legal moves
                game_winner = -1 if env.current_player == 1 else 1
                print(f"Game {ep+1}: Player {game_winner} wins - no legal moves available")
                break

            # Store current player (who's about to move)
            player_turn = env.current_player

            # Select model based on current player
            current_model = model1 if player_turn == 1 else model2

            # Try to get model prediction and select best move
            try:
                # Convert state to tensor with shape (1,1,8,8)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    q_values = current_model(state_tensor)
                
                # Get best move based on Q-values
                legal_q_values = {move: q_values[0, move_index(move)].item() for move in legal_moves}
                best_move = max(legal_q_values.items(), key=lambda x: x[1])[0]
                
                # Apply limited exploration during evaluation
                if random.random() < exploration_rate:
                    action = random.choice(legal_moves)
                    print(f"Game {ep+1}: Player {player_turn} taking random move: {action}")
                else:
                    action = best_move
            except Exception as e:
                print(f"Error selecting move: {e}")
                action = random.choice(legal_moves)  # Fallback to random move
                print(f"Game {ep+1}: Player {player_turn} taking fallback random move: {action}")

            # Take the selected action
            next_state, reward, done, info = env.step(action)
            move_count += 1

            # Track reward for the player who made the move
            if player_turn == 1:
                ep_reward_p1 += reward
            else:
                ep_reward_p2 += reward
                
            state = next_state

        # Record results for this episode
        total_rewards_p1.append(ep_reward_p1)
        total_rewards_p2.append(ep_reward_p2)
        total_moves.append(move_count)
        
        # Determine winner if not already determined
        if game_winner is None:
            if move_count >= 200 and not done:
                # Draw due to move limit
                draws += 1
                outcome = "DRAW (move limit)"
            elif done:
                # Game ended normally
                if ep_reward_p1 > ep_reward_p2:
                    wins_p1 += 1
                    outcome = "Player 1 WINS"
                    game_winner = 1
                elif ep_reward_p2 > ep_reward_p1:
                    wins_p2 += 1
                    outcome = "Player 2 WINS"
                    game_winner = -1
                else:
                    draws += 1
                    outcome = "DRAW"
            else:
                # This should not happen but just in case
                draws += 1
                outcome = "UNDEFINED RESULT"
        else:
            # Winner was already determined (no legal moves)
            if game_winner == 1:
                wins_p1 += 1
                outcome = "Player 1 WINS"
            else:
                wins_p2 += 1
                outcome = "Player 2 WINS"

        # Print detailed results for this game
        print(f"Game {ep+1}/{num_episodes}: {outcome} - Moves: {move_count}, P1 Reward: {ep_reward_p1:.1f}, P2 Reward: {ep_reward_p2:.1f}")
    
    # Calculate statistics
    total_time = time.time() - start_time
    avg_reward_p1 = np.mean(total_rewards_p1)
    avg_reward_p2 = np.mean(total_rewards_p2)
    avg_moves = np.mean(total_moves)
    win_rate_p1 = wins_p1 / num_episodes * 100
    win_rate_p2 = wins_p2 / num_episodes * 100
    draw_rate = draws / num_episodes * 100
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Games: {num_episodes}")
    print(f"Time: {total_time:.2f} seconds ({total_time/num_episodes:.2f} sec/game)")
    print("\nResults:")
    print(f"Player 1 Wins: {wins_p1} ({win_rate_p1:.1f}%)")
    print(f"Player 2 Wins: {wins_p2} ({win_rate_p2:.1f}%)")
    print(f"Draws: {draws} ({draw_rate:.1f}%)")
    print("\nPerformance:")
    print(f"Avg Reward P1: {avg_reward_p1:.2f}")
    print(f"Avg Reward P2: {avg_reward_p2:.2f}")
    print(f"Avg Moves per Game: {avg_moves:.2f}")
    print("="*50)

    # Create visualizations
    try:
        # Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')
            
        # Plot rewards per episode
        plt.figure(figsize=(12, 6))
        plt.plot(total_rewards_p1, label='Player 1 (Red)', color='red')
        plt.plot(total_rewards_p2, label='Player 2 (White)', color='blue')
        plt.xlabel('Game Number')
        plt.ylabel('Total Reward')
        plt.title('Rewards per Game')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('results/game_rewards.png')
        
        # Plot win distribution
        plt.figure(figsize=(10, 6))
        labels = ['Player 1 (Red)', 'Player 2 (White)', 'Draws']
        values = [wins_p1, wins_p2, draws]
        colors = ['red', 'blue', 'gray']
        plt.bar(labels, values, color=colors)
        plt.ylabel('Number of Games')
        plt.title('Game Outcomes')
        
        # Add percentage labels
        for i, v in enumerate(values):
            plt.text(i, v + 0.5, f"{v} ({v/num_episodes*100:.1f}%)", 
                     ha='center', va='bottom', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig('results/game_outcomes.png')
        
        # Show plots
        plt.show()
        
        print("Visualizations saved to 'results' directory")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        
    return {
        'wins_p1': wins_p1,
        'wins_p2': wins_p2,
        'draws': draws,
        'avg_reward_p1': avg_reward_p1,
        'avg_reward_p2': avg_reward_p2,
        'avg_moves': avg_moves
    }

if __name__ == '__main__':
    env = CheckersEnv()
    output_dim = 256  # Must match the move mapping in train.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create models
    q_net1 = DuelingQNetwork(output_dim).to(device)
    q_net2 = DuelingQNetwork(output_dim).to(device)
    
    # Try multiple model files with better error handling
    model_files = [
        "final_checker_modelT.pth", 
        "best_checker_model.pth",
        "final_checker_model.pth",
        "trained_checkers_model_final.pth"
    ]
    
    # Load first model (Player 1)
    model1_loaded = False
    model_file = "final_checker_modelT.pth"
    try:
        print(f"Trying to load Player 1 model from: {model_file}")
        q_net1.load_state_dict(torch.load(model_file, map_location=device))
        q_net1.eval()
        print(f"✓ Player 1 model loaded successfully from {model_file}")
        model1_loaded = True
        
    except Exception as e:
        print(f"✗ Failed to load from {model_file}: {e}")
    
    if not model1_loaded:
        print("WARNING: Could not load any model for Player 1. Using random play.")
        
        # Create a random play function
        def random_play1(x):
            return torch.rand((x.size(0), output_dim), device=device)
            
        # Replace the forward method
        q_net1.forward = random_play1
    
    # Load second model (Player 2)
    model2_loaded = False
    model_file = "final_checker_modelM.pth"
    
    try:
        print(f"Trying to load Player 2 model from: {model_file}")
        q_net2.load_state_dict(torch.load(model_file, map_location=device))
        q_net2.eval()
        print(f"✓ Player 2 model loaded successfully from {model_file}")
        model2_loaded = True
        
    except Exception as e:
        print(f"✗ Failed to load from {model_file}: {e}")
    
    if not model2_loaded:
        print("WARNING: Could not load any model for Player 2. Using random play.")
        
        # Create a random play function
        def random_play2(x):
            return torch.rand((x.size(0), output_dim), device=device)
            
        # Replace the forward method
        q_net2.forward = random_play2
    
    # Run evaluation with specified parameters
    print("\nStarting evaluation...")
    results = evaluate_model(
        env=env,
        model1=q_net1,
        model2=q_net2,
        num_episodes=50,  # Reduced number for faster testing
        exploration_rate=0.05  # Small chance of random moves
    )
