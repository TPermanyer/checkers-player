import gym
import numpy as np
from gym import spaces

class CheckersEnv(gym.Env):

    metadata = {"render.modes": ["human"]}
    
    def __init__(self):
        super(CheckersEnv, self).__init__()
        self.board_size = 8
        # The observation is the board state.(from -2 to 2 possible values on the board, 8x8 board)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.board_size, self.board_size), dtype=np.int8)
        # Action: (from_row, from_col, to_row, to_col)
        self.action_space = spaces.MultiDiscrete([self.board_size, self.board_size, self.board_size, self.board_size])
        self.reset()

    def reset(self):
        # Initialize the board with the starting positions.
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
    
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i + j) % 2 == 1:
                    if i < 3:
                        self.board[i, j] = -1  # player 2
                    elif i > 4:
                        self.board[i, j] = 1   # player 1
        self.current_player = 1  # Let player 1 start (adjust if needed)
        self.done = False
        return self.board.copy()

    def step(self, action):
        #executes an action
        #copy to not modify the original board
        if self.done:
            return self.board.copy(), 0, True, {}
        
        
        legal_moves = self.get_legal_moves(self.current_player)
        # Terminal condition: no legal moves available.
        if not legal_moves:
            self.done = True
            return self.board.copy(), -1, self.done, {"info": "No legal moves available."}
        
        if action not in legal_moves:
            # Illegal move: assign a penalty and end the episode.
            self.done = True
            return self.board.copy(), -5, self.done, {"info": "Illegal move attempted."}
        
        # Execute the move.
        self.make_move(action, self.current_player)
        
        # Check for win: if the opponent has no pieces left.
        if not self.has_pieces(-self.current_player):
            self.done = True
            return self.board.copy(), 10, self.done, {"info": "Win!"}
        
        # Switch turns.
        self.current_player *= -1
        return self.board.copy(), 0, self.done, {}

    def render(self, mode='human'):
        """Simple console rendering."""
        symbol_map = {
            0: '.',
            1: 'm',   # player 1 man
            -1: 'o',  # player 2 man
            2: 'M',   # player 1 king
            -2: 'O'   # player 2 king
        }
        print("Current player:", self.current_player)
        for row in self.board:
            print(" ".join(symbol_map[val] for val in row))
        print()

    def get_legal_moves(self, player):
        """
        Returns a list of legal moves for the given player.
        Each move is a tuple: (from_row, from_col, to_row, to_col).
        Capturing moves are forced if available.
        """
        moves = []
        capture_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                piece = self.board[i, j]
                if piece == 0:
                    continue
                if (player == 1 and piece > 0) or (player == -1 and piece < 0):
                    # Determine move directions.
                    if abs(piece) == 1:  # man
                        directions = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
                    else:
                        # King can move in all four diagonals.
                        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    
                    for dr, dc in directions:
                        new_r, new_c = i + dr, j + dc
                        # Check simple move.
                        if 0 <= new_r < self.board_size and 0 <= new_c < self.board_size:
                            if self.board[new_r, new_c] == 0:
                                moves.append((i, j, new_r, new_c))
                        # Check for capture: landing cell must be empty.
                        cap_r, cap_c = i + 2 * dr, j + 2 * dc
                        if (0 <= new_r < self.board_size and 0 <= new_c < self.board_size and
                            0 <= cap_r < self.board_size and 0 <= cap_c < self.board_size):
                            opponent = -player
                            adjacent = self.board[new_r, new_c]
                            if adjacent == opponent or adjacent == (2 if opponent == 1 else -2):
                                if self.board[cap_r, cap_c] == 0:
                                    capture_moves.append((i, j, cap_r, cap_c))
        return capture_moves if capture_moves else moves

    def make_move(self, action, player):
        """
        Executes a move on the board.
        Removes an opponent's piece if a capture is made.
        """
        from_row, from_col, to_row, to_col = action
        piece = self.board[from_row, from_col]
        self.board[to_row, to_col] = piece
        self.board[from_row, from_col] = 0
        
        if abs(to_row - from_row) == 2:
            jumped_r = (from_row + to_row) // 2
            jumped_c = (from_col + to_col) // 2
            self.board[jumped_r, jumped_c] = 0
        
        # Promotion to king.
        if player == 1 and to_row == 0 and piece == 1:
            self.board[to_row, to_col] = 2
        elif player == -1 and to_row == self.board_size - 1 and piece == -1:
            self.board[to_row, to_col] = -2

    def has_pieces(self, player):
        """Return True if the player has any pieces left on the board."""
        if player == 1:
            return np.any(self.board == 1) or np.any(self.board == 2)
        else:
            return np.any(self.board == -1) or np.any(self.board == -2)