"""
A minimax player for Quixo
"""

from game import Game, Player, Move
from copy import deepcopy
from tqdm.auto import tqdm
import numpy as np

class GameWrapper(Game):
    def __init__(self, game : Game) -> None:
        super().__init__() #useless?
        self._board = game.get_board()
        self.current_player_idx = game.get_current_player()

    def make_move(self, from_pos: tuple[int, int], move: Move, player: int) -> None:
        '''Perform a move'''
        if player not in (0, 1):
            return
        prev_value = deepcopy(self._board[(from_pos[1], from_pos[0])])
        acceptable = self.take((from_pos[1], from_pos[0]), player)
        if acceptable:
            acceptable = self.slide((from_pos[1], from_pos[0]), move)
            if not acceptable:
                # restore previous
                self._board[(from_pos[1], from_pos[0])] = prev_value
        return acceptable
    
    def check_sequences(self) -> tuple[list[int], list[int]]:
        """Check the number of adjacent 2, 3, 4 and 5 pieces for each player."""

        # 1. Initialize the sequences
        x_sequences = [0, 0, 0, 0]
        o_sequences = [0, 0, 0, 0]

        # 2. Check the rows
        for row in self._board:
            # 2.1. Initialize the counters
            x_count = 0
            o_count = 0
            # 2.2. Check the pieces in the row
            for piece in row:
                # 2.2.1. If the piece belongs to player 0
                if piece == 0:
                    #
                    x_count += 1
                    o_count = 0
                # 2.2.2. If the piece belongs to player 1
                elif piece == 1:
                    x_count = 0
                    o_count += 1
                # 2.2.3. If the piece is neutral
                else:
                    x_count = 0
                    o_count = 0
                # 2.2.4. Update the sequences
                if x_count > 1: x_sequences[x_count-2] += 1
                if o_count > 1: o_sequences[o_count-2] += 1
        
        # 3. Check the columns
        for col in range(5):
            # 3.1. Initialize the counters
            x_count = 0
            o_count = 0
            # 3.2. Check the pieces in the column
            for row in range(5):
                # 3.2.1. Get the piece
                piece = self._board[row, col]
                # 3.2.2. If the piece belongs to player 0
                if piece == 0:
                    x_count += 1
                    o_count = 0
                # 3.2.3. If the piece belongs to player 1
                elif piece == 1:
                    x_count = 0
                    o_count += 1
                # 3.2.4. If the piece is neutral
                else:
                    x_count = 0
                    o_count = 0
                # 3.2.5. Update the sequences
                if x_count > 1: x_sequences[x_count-2] += 1
                if o_count > 1: o_sequences[o_count-2] += 1

        # 4. Check the principal diagonal
        # 4.1. Initialize the counters
        x_count = 0
        o_count = 0
        # 4.2. Check the pieces in the diagonal
        for i in range(5):
            # 4.2.1. Get the piece
            piece = self._board[i, i]
            # 4.2.2. If the piece belongs to player 0
            if piece == 0:
                x_count += 1
                o_count = 0
            # 4.2.3. If the piece belongs to player 1
            elif piece == 1:
                x_count = 0
                o_count += 1
            # 4.2.4. If the piece is neutral
            else:
                x_count = 0
                o_count = 0
            # 4.2.5. Update the sequences
            if x_count > 1: x_sequences[x_count-2] += 1
            if o_count > 1: o_sequences[o_count-2] += 1

        # 5. Check the secondary diagonal
        # 5.1. Initialize the counters
        x_count = 0
        o_count = 0
        # 5.2. Check the pieces in the diagonal
        for i in range(5):
            # 5.2.1. Get the piece
            piece = self._board[i, -(i+1)]
            # 5.2.2. If the piece belongs to player 0
            if piece == 0:
                x_count += 1
                o_count = 0
            # 5.2.3. If the piece belongs to player 1
            elif piece == 1:
                x_count = 0
                o_count += 1
            # 5.2.4. If the piece is neutral
            else:
                x_count = 0
                o_count = 0
            # 5.2.5. Update the sequences
            if x_count > 1: x_sequences[x_count-2] += 1
            if o_count > 1: o_sequences[o_count-2] += 1

        # 6. Return the sequences
        return x_sequences, o_sequences


    def take(self, from_pos: tuple[int, int], player_id: int) -> bool:
        """Checks that {from_pos} is in the border and marks the cell with {player_id}"""
        row, col = from_pos
        from_border = row in (0, 4) or col in (0, 4)
        if not from_border:
            return False  # the cell is not in the border
        if self._board[from_pos] != player_id and self._board[from_pos] != -1:
            return False  # the cell belongs to the opponent
        self._board[from_pos] = player_id
        return True
    
    def get_possible_actions(self) -> list[tuple[tuple[int, int], Move]]:
        # 1. Initialize the list of possible actions
        possible_actions = []

        # 2. Check the edges of the board for possible actions
        for i in range(5):

            # 2.1. Check TOP and BOTTOM edges
            for row in [0, 4]:
                # 2.1.1. Get the piece at the current position
                piece = self._board[row, i]
                # 2.1.2. If the piece is neutral or belongs to the current player
                if piece == -1 or piece == self.current_player_idx:
                    # 2.1.2.1. Add the option to move it from the opposite edge
                    if row == 0: possible_actions.append(((i, row), Move.BOTTOM))
                    else: possible_actions.append(((i, row), Move.TOP))
                    # 2.1.2.2. If the piece is not in a corner, also add the option to move it from parallel edges
                    if i != 0 and i != 4:
                        possible_actions.append(((i, row), Move.LEFT))
                        possible_actions.append(((i, row), Move.RIGHT))

            # 2.2. Check LEFT and RIGHT edges
            for col in [0, 4]:
                # 2.2.1. Get the piece at the current position
                piece = self._board[i, col]
                # 2.2.2. If the piece is neutral or belongs to the current player
                if piece == -1 or piece == self.current_player_idx:
                    # 2.2.2.1. Add the option to move it from the opposite edge
                    if col == 0: possible_actions.append(((col, i), Move.RIGHT))
                    else: possible_actions.append(((col, i), Move.LEFT))
                    # 2.2.2.2. If the piece is not in a corner, also add the option to move it from parallel edges
                    if i != 0 and i != 4:
                        possible_actions.append(((col, i), Move.TOP))
                        possible_actions.append(((col, i), Move.BOTTOM))

        # 3. Return the list of possible actions
        return possible_actions
    

    def slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        '''Slide the other pieces'''
        if slide not in self.__acceptable_slides(from_pos):
            return False  # consider raise ValueError('Invalid argument value')
        axis_0, axis_1 = from_pos
        # np.roll performs a rotation of the element of a 1D ndarray
        if slide == Move.RIGHT:
            self._board[axis_0] = np.roll(self._board[axis_0], -1)
        elif slide == Move.LEFT:
            self._board[axis_0] = np.roll(self._board[axis_0], 1)
        elif slide == Move.BOTTOM:
            self._board[:, axis_1] = np.roll(self._board[:, axis_1], -1)
        elif slide == Move.TOP:
            self._board[:(axis_0 + 1), axis_1] = np.roll(self._board[:(axis_0 + 1), axis_1], 1)
        return True
    
    @staticmethod
    def __acceptable_slides(from_position: tuple[int, int]):
        """When taking a piece from {from_position} returns the possible moves (slides)"""
        acceptable_slides = [Move.BOTTOM, Move.TOP, Move.LEFT, Move.RIGHT]
        axis_0 = from_position[0]    # axis_0 = 0 means uppermost row
        axis_1 = from_position[1]    # axis_1 = 0 means leftmost column

        if axis_0 == 0:  # can't move upwards if in the top row...
            acceptable_slides.remove(Move.TOP)
        elif axis_0 == 4:
            acceptable_slides.remove(Move.BOTTOM)

        if axis_1 == 0:
            acceptable_slides.remove(Move.LEFT)
        elif axis_1 == 4:
            acceptable_slides.remove(Move.RIGHT)
        return acceptable_slides

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


