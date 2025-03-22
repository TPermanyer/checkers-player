import gym
import numpy as np
from gym import spaces

class CheckersEnv(gym.Env):

    metadata = {"render.modes": ["human"]}
    
    def __init__(self):
        super(CheckersEnv, self).__init__()
        self.board_size = 8
        # The observation is the board state (values from -2 to 2, 8x8 board)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self.board_size, self.board_size), dtype=np.int8)
        # Action: (from_row, from_col, to_row, to_col)
        self.action_space = spaces.MultiDiscrete([self.board_size, self.board_size, self.board_size, self.board_size])
        self.max_moves = 40  # Maximum allowed moves per game
        self.reset()



    def reset(self):
        #Initialize the board with 0s

        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)

        #Starting positions for player 1 and player 2
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i + j) % 2 == 1:
                    if i < 3:
                        self.board[i, j] = -1  # player 2
                    elif i > 4:
                        self.board[i, j] = 1   # player 1
        self.current_player = 1  #Player 1 starts

        self.done = False
        #Count pieces for each player
        self.prev_piece_counts = {
            1: self.get_piece_count(1),
            -1: self.get_piece_count(-1)
        }

        self.move_count = 0  # Initialize move counter
        return self.board.copy()
    

    def get_piece_count(self, player):
        """Returns the number of pieces for the given player."""
        if player == 1:
            return np.sum(self.board == 1) + np.sum(self.board == 2)
        else:
            return np.sum(self.board == -1) + np.sum(self.board == -2)


    def step(self, action):
        #FINISH
        if self.done:
            return self.board.copy(), 0, True, {}
        
        #Save pre-move piece count.
        prev_counts = {
            1: self.get_piece_count(1),
            -1: self.get_piece_count(-1)
        }
        
        # Track positions before the move for strategic rewards
        prev_board = self.board.copy()
        legal_moves, has_captures = self.get_legal_moves(self.current_player)
        
        
        
        # Execute the move
        self.make_move(action, self.current_player)
        
        # Initialize reward components

        reward = 0
        capture_bonus = 0
        
        # Track captures in this sequence
        captures_in_sequence = 0
        
        # Checks if a capture was made
        capture_made = (abs(action[2] - action[0]) == 2)

        if capture_made:
            captures_in_sequence += 1
            self.move_count = 0
            capture_bonus += 50  

            # Check if this specific piece can capture again
            can_capture_again = False
            # Get the new position of the piece after the move
            new_row, new_col = action[2], action[3]
            
            # Check specifically for multiple captures with the same piece
            multiple_capture_moves = []
            for move in self.get_legal_moves(self.current_player)[0]:
                if move[0] == new_row and move[1] == new_col and abs(move[2] - move[0]) == 2:
                    multiple_capture_moves.append(move)
                    can_capture_again = True
            
            # If this piece can capture again, return without switching players
            if can_capture_again:
                # Apply a bonus reward for the potential of a multiple capture
                capture_bonus += 25  # Additional bonus for potential chain capture
                
                # Return with the current rewards, maintaining the same player's turn
                new_counts = {
                    1: self.get_piece_count(1),
                    -1: self.get_piece_count(-1)
                }
                
                enemy_captured = prev_counts[-self.current_player] - new_counts[-self.current_player]
                reward = capture_bonus + (100 * enemy_captured)
                
                return self.board.copy(), reward, self.done, {
                    "multiple_capture": True, 
                    "legal_moves": multiple_capture_moves,
                    "captures_so_far": captures_in_sequence
                }
        
        # Get updated piece counts after the move
        new_counts = {
            1: self.get_piece_count(1),
            -1: self.get_piece_count(-1)
        }
        
        # Condiciones terminales:
        # Si el jugador que acaba de mover pierde todas sus piezas:
        if not self.has_pieces(self.current_player):
            self.done = True
            base_reward = -10000 
            print("Player has no pieces") 
        # Si el oponente no tiene piezas:
        elif not self.has_pieces(-self.current_player):
            self.done = True
            base_reward = +10000
            # Add bonus for winning with more remaining pieces
            base_reward += 10 * new_counts[self.current_player]
        else:
            base_reward = 0
        
        
        # Check if opponent has legal moves
        op_legal_moves, op_has_captures = self.get_legal_moves(-self.current_player)
        
        if not op_legal_moves:
            self.done = True
            return self.board.copy(), reward + 1000, self.done, {"info": "No legal moves available for opponent."}
        if op_has_captures:
            penalty = -100
        else:
            penalty = 0

        
        # Recompensa por capturar piezas enemigas.
        enemy_captured = prev_counts[-self.current_player] - new_counts[-self.current_player]
        capture_reward = 100 * enemy_captured
        
        # Check for king creation
        from_row, from_col, to_row, to_col = action
        if abs(prev_board[from_row, from_col]) == 1:  # Was a regular piece
            if abs(self.board[to_row, to_col]) == 2:  # Now it's a king
                capture_reward += 75  # Bonus for getting a king
        
        # Recompensa total.
        reward = base_reward + penalty + capture_reward + capture_bonus
        
        
        # Incrementa el contador de movimientos y verifica el límite.
        self.move_count += 1
        if self.move_count >= self.max_moves:
            self.done = True
            reward -= 10  # Fixed missing assignment (was 'reward -10')
        
        
        
        # Si no se cumple la condición anterior, se cambia de turno.
        self.current_player *= -1
        self.prev_piece_counts = new_counts.copy()
        
        return self.board.copy(), reward, self.done, {}

    
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

        return capture_moves if capture_moves else moves, True if capture_moves else False

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
