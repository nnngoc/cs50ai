"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = 0
    o_count = 0

    for row in board:
        x_count += row.count(X)
        o_count += row.count(O)
    
    # in the initial game state, X get the first move
    if x_count <= o_count:
        return X 
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # All possible action
    possible_actions = set()

    for i , row in enumerate(board):
        for j, items in enumerate(row):
            if items is None:
                possible_actions.add((i, j))

    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Copy a new board
    new_board = deepcopy(board)

    i, j = action

    if board[i][j] is not None:
        raise Exception
    else:
        new_board[i][j] = player(board)
    
    return new_board



def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for player in (X, O):
        # Set array won the game
        array_win = [player] * 3
        # Check move horizontal
        for i in board:
            if i == array_win:
                return player
        
        # Check move vertical
        for j in range(3):
            column = []
            for c in range(3):
                column.append(board[c][j])
            if column == array_win:
                return player

        # Check move diagonal
        diagonal_1 = []
        diagonal_2 = []
        for k in range(3):
            diagonal_1.append(board[k][k])
            diagonal_2.append(board[k][~k])
        if diagonal_1 == array_win or diagonal_2 == array_win:
            return player
        
    return None
 
            
def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # The game is over
    if winner(board) is not None:
        return True
    
    # The game is still in progress
    for row in board:
        if EMPTY in row:
            return False
    
    # No possible move
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    # X won the game
    if winner(board) == X:
        return 1
    # O won the game
    elif winner(board) == O:
        return -1
    # Game has ended in a tie
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    def max_val(board):
        optimal = ()
        if terminal(board):
            return utility(board), optimal
        else:
            v = -math.inf
            for action in actions(board):
                min = min_val(result(board,action))[0]
                if min > v:
                    v = min
                    optimal = action
            return v, optimal
    
    def min_val(board):
        optimal = ()
        if terminal(board):
            return utility(board), optimal
        else:
            v = math.inf
            for action in actions(board):
                max = max_val(result(board, action))[0]
                if max < v:
                    v = max
                    optimal = action
            return v, optimal
    
    # return None if board is terminal
    if terminal(board):
        return None
    # return optimal move for X
    if player(board) == X:
        return max_val(board)[1]
    # return optimal move for O
    else:
        return min_val(board)[1]