import chess
import numpy as np


def is_terminal(board):
    outcome = board.outcome()
    
    if outcome is None: # if outcome is None the game is still going; i.e. non-is_terminal state
        # print('game has not ended yet')
        return None
    
    else: # is_terminal states
        winner = outcome.winner # True for WHITE player and False for BLACK player
        termination = outcome.termination
    
        if termination == chess.Termination.CHECKMATE:
            if winner:
                # print('White player has won')
                return 1
            else:
                # print('Black player has won')
                return -1
            
        else: # for other outcomes, the game is drawn
            # print('the game is draw')
            return 0
            
            
def heuristic_chess(board):
    terminal = is_terminal(board)

    if terminal is not None:
        return terminal
    
    else:
        f_pawn = cal_Fpiece(board,chess.PAWN,chess.WHITE,chess.BLACK)
        f_knight = cal_Fpiece(board,chess.KNIGHT,chess.WHITE,chess.BLACK)
        f_bishop = cal_Fpiece(board,chess.BISHOP,chess.WHITE,chess.BLACK)
        f_rook = cal_Fpiece(board,chess.ROOK,chess.WHITE,chess.BLACK)
        f_queen = cal_Fpiece(board,chess.QUEEN,chess.WHITE,chess.BLACK)
        
        heuristic_value = (f_pawn + 3 * f_knight + 4 * f_bishop + 5 * f_rook + 9 * f_queen) / 100
        return heuristic_value



def cal_Fpiece(board,Piece,White,Black):
    return len(board.pieces(Piece,White))-len(board.pieces(Piece,Black))




def is_cutoff(board, current_depth, depth_limit = 2):
    
    terminal = is_terminal(board)
    return terminal is not None or current_depth == depth_limit 
    
    
    
def h_minimax(board, depth_limit=2):
    
    return max_node(board, 0, depth_limit)



def max_node(board, current_depth, depth_limit):
    
    if is_cutoff(board, current_depth, depth_limit):
        return heuristic_chess(board), None
    
    v, move = -np.infty, -np.infty
    for a in  board.legal_moves:
        
        board.push(a)
        v2, a2 = min_node(board, current_depth + 1, depth_limit)
        if v2 > v:
            v, move = v2, a
            
        board.pop()
    return v, move 
            


def min_node(board, current_depth, depth_limit):
    
    if is_cutoff(board, current_depth, depth_limit):
        return heuristic_chess(board), None
    
    v, move = np.infty, np.infty
    for a in  board.legal_moves:
        
        board.push(a)
        v2, a2 = max_node(board, current_depth + 1, depth_limit)
        if v2 < v:
            v, move = v2, a
            
        board.pop()
    return v, move 



def h_minimax_alpha_beta(board, depth_limit = 2):
    
    return max_node_ab(board, 0, depth_limit, -np.infty, np.infty)



def max_node_ab(board, current_depth, depth_limit, alpha, beta):
    
    if is_cutoff(board, current_depth, depth_limit):
        return heuristic_chess(board), None
    
    v, move = -np.infty, -np.infty
    for a in  board.legal_moves:
        
        board.push(a)
        v2, a2 = min_node_ab(board, current_depth + 1, depth_limit, alpha, beta)
        if v2 > v:
            v, move = v2, a
            alpha = max(v, alpha)
            
        board.pop()
        
        if v >= beta:
            return v, move
            
    return v, move 
            


def min_node_ab(board, current_depth, depth_limit, alpha, beta):
    
    if is_cutoff(board, current_depth, depth_limit):
        return heuristic_chess(board), None
    
    v, move = np.infty, np.infty
    for a in  board.legal_moves:
        
        board.push(a)
        v2, a2 = max_node_ab(board, current_depth + 1, depth_limit, alpha, beta)
        if v2 < v:
            v, move = v2, a
            beta = min(v, beta)
            
        board.pop()
            
        if v <= alpha:
            return v, move
            
    return v, move 
