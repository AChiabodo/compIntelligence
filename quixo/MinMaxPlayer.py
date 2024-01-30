"""
A minimax player for Quixo
"""

from game import Game, Player, Move
from copy import deepcopy
from tqdm.auto import tqdm
import numpy as np
from GameWrapper import GameWrapper

class MinmaxPlayer(Player):

    def __init__(self, depth: int=2) -> None:
        """
          Initializes the player

          - `depth`: The maximum number of levels or moves ahead 
            that the algorithm explores in the game tree to make its decision.
        """
        self.depth = depth
        self.verbose = False
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        """Returns the best action for the current player using minimax algorithm"""
        #print("Thinking...")
        # 1. Get the current player and save it as the maximizing player
        minimax_player_id = game.get_current_player()
        maximizing_player = True

        # 2. Call minimax recursively for the current player
        _, best_action = self.minimax(game, self.depth, maximizing_player, minimax_player_id)
        
        return best_action
    
    def minimax(self, game: 'GameWrapper', depth: int, maximizing_player: bool, minimax_player_id: int,alpha_value = float('-inf'), beta_value = float('inf')) -> tuple[int, tuple[tuple[int, int], Move]]:
        """
        Minimax recursive algorithm for Quixo
        """
        if depth != 0 and self.verbose:
            print("minimax depth ", depth, " ", maximizing_player, " ", minimax_player_id)
        game = GameWrapper(game)
        alpha = alpha_value
        beta = beta_value
        # 0. TERMINAL CONDITIONS
        # If we have reached the maximum depth or the game is over
        winner = game.check_winner()
        if depth == 0 or winner != -1:
            # Return the heuristic value of the game state
            return self.evaluate(game, winner, minimax_player_id), None

        # 1. Get the possible actions for the current player
        possible_actions = game.get_possible_actions()

        # 2. If the current player is the maximizing player
        if maximizing_player:
            
            # 2.1. Initialize the best value and the best action
            best_value = float('-inf')
            best_action = None

            # 2.2. For each possible action
            for i , (from_pos, slide) in enumerate(possible_actions):
                if depth != 0 and self.verbose: 
                    print(f"Action {i} of {len(possible_actions)}")
                # 2.2.1. Make a copy of the game and perform the action
                game_copy : GameWrapper = deepcopy(game)
                game_copy.make_move(from_pos, slide, minimax_player_id)

                # 2.2.2. Call minimax recursively for the opponent
                value, _ = self.minimax(game_copy, depth - 1, not maximizing_player, minimax_player_id, alpha, beta)

                # 2.2.3. If the value is better than the current best value
                if value > best_value:
                    best_value = value
                    best_action = (from_pos, slide)

                # 2.2.4. Update alpha
                alpha = max(alpha, best_value)

                # 2.2.5. Check if we can prune
                if beta <= alpha:
                    break

            # 2.3. Return the best value and the best action
            return best_value, best_action
        
        # 3. If the current player is the minimizing player
        else:
            
            # 3.1. Initialize the best value and the best action
            best_value = float('inf')
            best_action = None

            # 3.2. For each possible action
            for i , (from_pos, slide) in enumerate(possible_actions):
                if depth != 0 and self.verbose: 
                    print(f"Action {i} of {len(possible_actions)}")
                # 3.2.1. Make a copy of the game and perform the action
                game_copy = deepcopy(game)
                game_copy.make_move(from_pos, slide, 1 - minimax_player_id)

                # 3.2.2. Call minimax recursively for the opponent
                value, _ = self.minimax(game_copy, depth - 1, not maximizing_player, minimax_player_id, alpha, beta)

                # 3.2.3. If the value is better than the current best value
                if value < best_value:
                    best_value = value
                    best_action = (from_pos, slide)
            
            # 3.3. Return the best value and the best action
            return best_value, best_action

    def evaluate(self, game: 'GameWrapper', winner: int, minimax_player_id: int) -> int:
        """
        Heuristic function for Quixo
        """
        # A. Get the value for a winning/losing game
        # ..........................................
        if winner != -1:
            if winner == minimax_player_id:
                return 100
            else:
                return -100
            
        # B. Get the value for a game that is not over
        # ............................................
            
        # 1. Get the sequences of the current player and the opponent
        x_sequences, o_sequences = game.check_sequences()

        # 2. Get the sequences of the current player and the opponent
        if minimax_player_id == 0:
            player_sequences = x_sequences
            opponent_sequences = o_sequences
        else:
            player_sequences = o_sequences
            opponent_sequences = x_sequences
        
        # 3. Get the total score
        player_score = player_sequences[0] + player_sequences[1] * 3 + player_sequences[2] * 5
        opponent_score = opponent_sequences[0] + opponent_sequences[1] * 3 + opponent_sequences[2] * 5
        return player_score - opponent_score


if __name__ == "__main__":
    player = MinmaxPlayer(depth=2)