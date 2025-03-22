import sys
import pygame
import time
import torch
import random
import argparse
import os
from environment import CheckersEnv
from train import DuelingQNetwork, move_index

# Constants for the graphical interface
WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Colors (RGB)
LIGHT_SQUARE = (232, 235, 239)
DARK_SQUARE = (125, 135, 150)
PLAYER1_COLOR = (255, 0, 0)      # Red for player 1
PLAYER2_COLOR = (255, 255, 255)  # White for player 2
KING_MARK = (255, 215, 0)        # Gold for king marker
HIGHLIGHT_COLOR = (0, 255, 0)    # Green for highlighting selected pieces
POSSIBLE_MOVE_COLOR = (173, 216, 230)  # Light blue for possible moves

def draw_board(screen, env, board, selected_piece=None, possible_moves=None):
    """Draws the checkers board and pieces based on the board state."""
    # Draw board squares
    for row in range(ROWS):
        for col in range(COLS):
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            
            # Highlight possible moves
            if possible_moves and (row, col) in [(m[2], m[3]) for m in possible_moves]:
                pygame.draw.rect(screen, POSSIBLE_MOVE_COLOR, 
                                (col * SQUARE_SIZE + 5, row * SQUARE_SIZE + 5, 
                                 SQUARE_SIZE - 10, SQUARE_SIZE - 10), 3)
    
    # Draw pieces
    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row][col]
            if piece != 0:
                center = (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2)
                if piece > 0:  # Player 1 (Red)
                    # Highlight selected piece
                    if selected_piece and selected_piece == (row, col):
                        pygame.draw.circle(screen, HIGHLIGHT_COLOR, center, SQUARE_SIZE // 2 - 5)
                    
                    pygame.draw.circle(screen, PLAYER1_COLOR, center, SQUARE_SIZE // 2 - 10)
                    if piece == 2:  # King piece for player 1
                        pygame.draw.circle(screen, KING_MARK, center, SQUARE_SIZE // 2 - 20)
                else:  # Player -1 (White)
                    # Highlight selected piece
                    if selected_piece and selected_piece == (row, col):
                        pygame.draw.circle(screen, HIGHLIGHT_COLOR, center, SQUARE_SIZE // 2 - 5)
                        
                    pygame.draw.circle(screen, PLAYER2_COLOR, center, SQUARE_SIZE // 2 - 10)
                    if piece == -2:  # King piece for player 2
                        pygame.draw.circle(screen, KING_MARK, center, SQUARE_SIZE // 2 - 20)
    
    # Display current player turn
    font = pygame.font.SysFont('Arial', 30)
    current_player_text = "Red's Turn" if env.current_player == 1 else "White's Turn"
    text_surface = font.render(current_player_text, True, (0, 0, 0))
    screen.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, 10))

def get_board_pos_from_mouse(pos):
    """Converts mouse (x, y) coordinates to board row and column indices."""
    x, y = pos
    col = x // SQUARE_SIZE
    row = y // SQUARE_SIZE
    return row, col

def handle_human_turn(event, state, selected_piece, legal_moves, env):
    """Handles a human player's turn."""
    done = False
    move_made = False
    
    if event.type == pygame.MOUSEBUTTONDOWN:
        pos = pygame.mouse.get_pos()
        row, col = get_board_pos_from_mouse(pos)
        
        # If no piece selected, try to select one
        if selected_piece is None:
            if 0 <= row < 8 and 0 <= col < 8 and state[row][col] != 0:
                # Check if piece belongs to current player
                if (env.current_player == 1 and state[row][col] > 0) or (env.current_player == -1 and state[row][col] < 0):
                    selected_piece = (row, col)
                    print(f"Selected piece at {selected_piece}")
        else:
            # Try to move the selected piece to the clicked square
            move = (selected_piece[0], selected_piece[1], row, col)
            
            if move in legal_moves:
                print(f"Player {env.current_player} move: {move}")
                current_turn = env.current_player
                state, reward, done, info = env.step(move)
                print(f"Reward: {reward} | Info: {info}")
                move_made = True
                
                # Check if it's a multiple capture (same player's turn)
                if env.current_player == current_turn:
                    selected_piece = (move[2], move[3])
                    print(f"Multiple capture available. Piece now at {selected_piece}")
                else:
                    selected_piece = None
            else:
                # If clicked on a different own piece, select that piece instead
                if 0 <= row < 8 and 0 <= col < 8 and state[row][col] != 0:
                    if (env.current_player == 1 and state[row][col] > 0) or (env.current_player == -1 and state[row][col] < 0):
                        selected_piece = (row, col)
                        print(f"Selected new piece at {selected_piece}")
                    else:
                        # Clicked on opponent's piece or empty square - deselect
                        selected_piece = None
                else:
                    # If the move is invalid, reset selection
                    selected_piece = None
    
    return state, selected_piece, done, move_made

def play_game_gui(env, q_net, mode="human_vs_ai", human_player=1):
    """
    Play a game of checkers with GUI.
    
    Args:
        env: The checkers environment
        q_net: The trained model (only used in human_vs_ai mode)
        mode: "human_vs_ai" or "human_vs_human"
        human_player: Which player the human controls in human_vs_ai mode (1 or -1)
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Checkers Game")
    clock = pygame.time.Clock()
    
    state = env.reset()
    selected_piece = None
    possible_moves = []
    running = True
    
    # Set device for model
    device = torch.device("cpu")
    if q_net is not None:
        device = next(q_net.parameters()).device
    
    # Main game loop
    while running:
        # Get legal moves for the current player
        legal_moves, has_captures = env.get_legal_moves(env.current_player)
        
        # Check for game over due to no legal moves
        if not legal_moves:
            winner = "Red" if env.current_player == -1 else "White"
            print(f"Game over! {winner} wins (no legal moves for opponent)")
            time.sleep(3)
            running = False
            break
        
        # Filter possible moves for the selected piece
        if selected_piece:
            possible_moves = [m for m in legal_moves if m[0] == selected_piece[0] and m[1] == selected_piece[1]]
        else:
            possible_moves = []
            
        # Draw the current game state
        draw_board(screen, env, state, selected_piece, possible_moves)
        pygame.display.flip()
        
        # Process events
        move_made = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return
            
            # Human vs Human mode - both players controlled by human
            if mode == "human_vs_human":
                state, selected_piece, done, move_made = handle_human_turn(event, state, selected_piece, legal_moves, env)
                if done:
                    winner = "Red" if env.current_player == 1 else "White"
                    print(f"Game over! {winner} wins!")
                    time.sleep(3)
                    running = False
                    break
            
            # Human vs AI mode - only handle human's turn
            elif mode == "human_vs_ai" and env.current_player == human_player:
                state, selected_piece, done, move_made = handle_human_turn(event, state, selected_piece, legal_moves, env)
                if done:
                    winner = "Red" if human_player == 1 else "White"
                    print(f"Game over! {winner} (You) wins!")
                    time.sleep(3)
                    running = False
                    break
        
        # AI's turn in human vs AI mode
        if mode == "human_vs_ai" and env.current_player != human_player and running and not move_made:
            # Add a delay to make AI moves visible
            pygame.time.delay(500)
            
            # Check if AI has legal moves
            legal_moves, has_captures = env.get_legal_moves(env.current_player)
            if not legal_moves:
                winner = "Red" if human_player == 1 else "White"
                print(f"Game over! {winner} (You) wins (AI has no legal moves)")
                time.sleep(3)
                running = False
                break
            
            # Convert board state to tensor for model input
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Get Q-values from the model
            with torch.no_grad():
                q_values = q_net(state_tensor)
            
            # Print Q-values for debugging
            print(f"Q-values shape: {q_values.shape}")
            
            # Select move with highest Q-value among legal moves
            legal_q_values = {}
            for move in legal_moves:
                try:
                    idx = move_index(move)
                    legal_q_values[move] = q_values[0, idx].item()
                except Exception as e:
                    print(f"Error calculating q-value for move {move}: {e}")
                    legal_q_values[move] = 0.0
            
            # If we need to force captures
            if has_captures:
                capture_moves = [m for m in legal_moves if abs(m[0] - m[2]) == 2]
                if capture_moves:
                    legal_q_values = {m: v for m, v in legal_q_values.items() if m in capture_moves}
            
            # Select best move
            if legal_q_values:
                move = max(legal_q_values.items(), key=lambda x: x[1])[0]
            else:
                # Fallback to random legal move if Q-values couldn't be calculated
                move = random.choice(legal_moves)
            
            print(f"AI move: {move}")
            
            # Execute AI move
            current_turn = env.current_player
            state, reward, done, info = env.step(move)
            print(f"Reward: {reward} | Info: {info}")
            
            # Handle multiple captures for AI
            while env.current_player == current_turn and not done:
                # AI has a multiple capture available
                legal_moves, _ = env.get_legal_moves(env.current_player)
                # Filter for moves starting from the current piece position
                next_moves = [m for m in legal_moves if m[0] == move[2] and m[1] == move[3] and abs(m[0] - m[2]) == 2]
                
                if not next_moves:
                    break
                
                # Select the best next capture
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = q_net(state_tensor)
                
                next_q_values = {}
                for next_move in next_moves:
                    try:
                        idx = move_index(next_move)
                        next_q_values[next_move] = q_values[0, idx].item()
                    except:
                        next_q_values[next_move] = 0.0
                
                if next_q_values:
                    move = max(next_q_values.items(), key=lambda x: x[1])[0]
                else:
                    move = random.choice(next_moves)
                
                print(f"AI multiple capture: {move}")
                pygame.time.delay(500)  # Delay to see the multiple capture
                
                # Update the display to show the intermediate state
                draw_board(screen, env, state, None, [])
                pygame.display.flip()
                
                # Execute the next capture
                state, reward, done, info = env.step(move)
                print(f"Multiple capture reward: {reward} | Info: {info}")
            
            if done:
                winner = "White" if human_player == 1 else "Red"
                print(f"Game over! {winner} (AI) wins!")
                time.sleep(3)
                running = False
        
        clock.tick(30)  # Limit to 30 FPS
    
    pygame.quit()

def show_menu():
    """Display a game mode selection menu."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Checkers - Game Mode Selection")
    
    font_title = pygame.font.SysFont('Arial', 40)
    font_option = pygame.font.SysFont('Arial', 30)
    
    title = font_title.render('Select Game Mode', True, (0, 0, 0))
    option1 = font_option.render('1. Human vs AI (play as Red)', True, (0, 0, 0))
    option2 = font_option.render('2. Human vs AI (play as White)', True, (0, 0, 0))
    option3 = font_option.render('3. Human vs Human', True, (0, 0, 0))
    
    running = True
    mode = None
    human_player = None
    
    while running:
        screen.fill((200, 200, 200))
        
        # Display title and options
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 150))
        screen.blit(option1, (WIDTH//2 - option1.get_width()//2, 300))
        screen.blit(option2, (WIDTH//2 - option2.get_width()//2, 350))
        screen.blit(option3, (WIDTH//2 - option3.get_width()//2, 400))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    mode = "human_vs_ai"
                    human_player = 1
                    running = False
                elif event.key == pygame.K_2:
                    mode = "human_vs_ai"
                    human_player = -1
                    running = False
                elif event.key == pygame.K_3:
                    mode = "human_vs_human"
                    human_player = None
                    running = False
    
    pygame.quit()
    return mode, human_player

def create_random_model(output_dim, device):
    """Create a model that returns random Q-values"""
    model = DuelingQNetwork(output_dim).to(device)
    
    # Override the forward method
    def random_forward(self, x):
        batch_size = x.size(0)
        return torch.rand((batch_size, output_dim), device=device)
    
    # Bind the random_forward method to the model instance
    import types
    model.forward = types.MethodType(random_forward, model)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play Checkers with GUI')
    parser.add_argument('--mode', choices=['human_vs_ai', 'human_vs_human'], help='Game mode')
    parser.add_argument('--player', type=int, choices=[1, -1], default=1, help='Which player human controls in human_vs_ai mode (1=red, -1=white)')
    parser.add_argument('--random_ai', action='store_true', help='Use random AI instead of trained model')
    args = parser.parse_args()
    
    env = CheckersEnv()
    
    # If mode not specified via command line, show menu
    if args.mode is None:
        mode, human_player = show_menu()
    else:
        mode = args.mode
        human_player = args.player if mode == "human_vs_ai" else None
    
    # Only load the model if we're playing against AI
    q_net = None
    if mode == "human_vs_ai":
        output_dim = 256  # Based on your move index mapping
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Use random AI if requested
        if args.random_ai:
            print("Using random AI (ignoring trained models)")
            q_net = create_random_model(output_dim, device)
        else:
            # Initialize the DuelingQNetwork
            q_net = DuelingQNetwork(output_dim).to(device)
            
            # Look for models in current directory
            model_files = [
                "best_checker_model.pth",
                "final_checker_model.pth",
                "best_checkers_model.pth",
                "trained_checkers_model_final.pth",
                "self_play_model_iter5.pth",
                "self_play_model_iter4.pth",
                "self_play_model_iter3.pth",
                "self_play_model_iter2.pth",
                "self_play_model_iter1.pth"
            ]
            
            # Add any .pth files in current directory
            for file in os.listdir('.'):
                if file.endswith('.pth') and file not in model_files:
                    model_files.append(file)
            
            model_loaded = False
            for model_file in model_files:
                if not os.path.exists(model_file):
                    print(f"File does not exist: {model_file}")
                    continue
                    
                try:
                    print(f"Attempting to load {model_file}...")
                    # Try direct state_dict loading
                    try:
                        q_net.load_state_dict(torch.load(model_file, map_location=device))
                        model_loaded = True
                        print(f"Model loaded successfully from {model_file}")
                        break
                    except Exception as direct_error:
                        print(f"Direct loading failed: {direct_error}")
                        # Try loading as checkpoint
                        checkpoint = torch.load(model_file, map_location=device)
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            q_net.load_state_dict(checkpoint['model_state_dict'])
                            model_loaded = True
                            print(f"Checkpoint loaded successfully from {model_file}")
                            break
                        else:
                            print(f"File is not a valid checkpoint: {model_file}")
                except Exception as e:
                    print(f"Failed to load {model_file}: {e}")
            
            if not model_loaded:
                print("Could not load any model. The AI will make random moves.")
                q_net = create_random_model(output_dim, device)
            
            # Set to evaluation mode
            q_net.eval()
    
    # Start the game with the selected mode
    play_game_gui(env, q_net, mode, human_player)
