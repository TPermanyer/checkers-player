import time
import torch
import random
from environment import CheckersEnv
from train import QNetwork  # Ensure QNetwork is importable from train.py

def play_game(env, q_net, human_player=1):
    state = env.reset()
    env.render()
    done = False
    
    while not done:
        legal_moves = env.get_legal_moves(env.current_player)
        if not legal_moves:
            print("No legal moves available. Game over.")
            break

        if env.current_player == human_player:
            print("Your legal moves:")
            for idx, move in enumerate(legal_moves):
                print(f"{idx}: {move}")
            try:
                move_idx = int(input("Enter the index of your move: "))
            except ValueError:
                print("Invalid input. Please enter an integer.")
                continue
            if move_idx < 0 or move_idx >= len(legal_moves):
                print("Invalid move index. Try again.")
                continue
            action = legal_moves[move_idx]
        else:
            # Model's turn: use the Q-network to select a move (placeholder: random choice).
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_net.eval()
            with torch.no_grad():
                q_values = q_net(state_tensor)
            action = random.choice(legal_moves)
            print("Model chooses move:", action)

        state, reward, done, info = env.step(action)
        env.render()
        print("Reward:", reward, "| Info:", info)
        time.sleep(1)

if __name__ == "__main__":
    env = CheckersEnv()
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    output_dim = 64  # Must match what was used during training.
    q_net = QNetwork(input_dim, output_dim)
    # Load the trained model's state dictionary.
    q_net.load_state_dict(torch.load("trained_q_net_state.pth", map_location=torch.device("cpu")))
    q_net.eval()
    play_game(env, q_net)
