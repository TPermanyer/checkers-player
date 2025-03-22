# Checkers-Player
A model trained to play checkers using CNN .

## Checkers
Checkers (also known as Draughts) is a two-player strategy board game played on an 8x8 board.

### Basics:
Players: 2 (traditionally Red vs. Black)
Pieces: Each player begins with 12 pieces placed on dark squares.
Objective: Capture all opponent pieces or block them so they can't move.
### Rules:
Pieces move diagonally forward onto unoccupied squares.
Capturing: If an opponent's piece is diagonally adjacent with an empty square immediately beyond, you must jump and capture it.
Multiple jumps are possible and mandatory if available.
Pieces reaching the opposite end become "kings" (can move diagonally forward and backward).
### Winning:
A player wins by capturing all opponent pieces or leaving them with no legal moves.

## Approach
I will use OpenAi gym library for RL training and PyGames for the graphic implementation also with CUDA I will use hardware acceleration to speed up training time .

## Environment
Board representation:
    -  0: empty cell
    -  1: player 1 man
    - -1: player 2 man
    -  2: player 1 king
    - -2: player 2 king

The action is represented as a tuple:
    (from_row, from_col, to_row, to_col)

## Agent


## Training


## Testing



