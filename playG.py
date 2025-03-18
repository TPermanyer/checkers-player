import sys
import pygame
import time
import torch
import random
from environment import CheckersEnv
from train import QNetwork  # Ensure QNetwork is importable from train.py

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

def draw_board(screen, board):
    """Draws the checkers board and pieces based on the board state."""
    # Draw squares
    for row in range(ROWS):
        for col in range(COLS):
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    # Draw pieces
    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row][col]
            if piece != 0:
                center = (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2)
                if piece > 0:
                    pygame.draw.circle(screen, PLAYER1_COLOR, center, SQUARE_SIZE // 2 - 10)
                    if piece == 2:  # King piece for player 1
                        pygame.draw.circle(screen, KING_MARK, center, SQUARE_SIZE // 2 - 20)
                else:
                    pygame.draw.circle(screen, PLAYER2_COLOR, center, SQUARE_SIZE // 2 - 10)
                    if piece == -2:  # King piece for player 2
                        pygame.draw.circle(screen, KING_MARK, center, SQUARE_SIZE // 2 - 20)

def get_board_pos_from_mouse(pos):
    """Converts mouse (x, y) coordinates to board row and column indices."""
    x, y = pos
    col = x // SQUARE_SIZE
    row = y // SQUARE_SIZE
    return row, col

def play_game_gui(env, q_net, human_player=1):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Checkers Bot")
    clock = pygame.time.Clock()
    
    state = env.reset()
    selected_piece = None  # Store selected piece (row, col) when human clicks
    running = True
    while running:
        draw_board(screen, state)
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            
            # Human player's turn: click to select piece and then destination.
            if env.current_player == human_player:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    row, col = get_board_pos_from_mouse(pos)
                    # If no piece is selected, try to select one.
                    if selected_piece is None:
                        if state[row][col] != 0:
                            if (human_player == 1 and state[row][col] > 0) or (human_player == -1 and state[row][col] < 0):
                                selected_piece = (row, col)
                                print("Selected piece at", selected_piece)
                    else:
                        # Attempt to move the selected piece to the clicked square.
                        move = (selected_piece[0], selected_piece[1], row, col)
                        legal_moves = env.get_legal_moves(env.current_player)
                        if move in legal_moves:
                            print("Human move:", move)
                            state, reward, done, info = env.step(move)
                            print("Reward:", reward, "| Info:", info)
                            selected_piece = None
                            if done:
                                print("Game over!")
                                running = False
                        else:
                            print("Illegal move or wrong destination.")
                            selected_piece = None  # Reset selection if move invalid.
            else:
                # Model's turn: for now, select a random legal move.
                pygame.time.delay(500)  # Brief delay so the human can see the board update.
                legal_moves = env.get_legal_moves(env.current_player)
                if not legal_moves:
                    print("Model has no legal moves. Game over!")
                    running = False
                    break
                # In a refined version, use q_net to select the best move.
                move = random.choice(legal_moves)
                print("Model move:", move)
                state, reward, done, info = env.step(move)
                print("Reward:", reward, "| Info:", info)
                if done:
                    print("Game over!")
                    running = False
        
        clock.tick(30)  # Limit to 30 FPS

    pygame.quit()

if __name__ == "__main__":
    env = CheckersEnv()
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    output_dim = 64  # Must match the dimension used during training.
    q_net = QNetwork(input_dim, output_dim)
    
    # Attempt to load the trained model state; if not available, the model will make random moves.
    try:
        q_net.load_state_dict(torch.load("trained_q_net_state.pth", map_location=torch.device("cpu")))
        q_net.eval()
    except Exception as e:
        print("Could not load trained model, using random moves for model. Error:", e)
    
    play_game_gui(env, q_net)