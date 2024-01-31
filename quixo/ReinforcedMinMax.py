from copy import deepcopy
import os
import numpy as np
import random

from game import Game, Move, Player
from GameWrapper import GameWrapper

def encode_board(board : np.ndarray) -> tuple[int,int]:
    board_1 = np.where(board != -1, 0, board).ravel()
    board_2 = np.where(board != 1, 0, board).ravel()
    value_1 = int(''.join([bin(byte)[2:].rjust(8, '0') for byte in np.packbits(np.where(board_1 == -1, 1, board_1))])[:25],2)
    value_2 = int(''.join([bin(byte)[2:].rjust(8, '0') for byte in np.packbits(np.where(board_2 == 1, 1, board_2))])[:25],2)    
    return (value_1, value_2)

def decode_board(board : tuple[int,int]) -> np.ndarray:
    value_1, value_2 = board
    binary_string_1 = format(value_1, 'b').zfill(25)
    binary_string_2 = format(value_2, 'b').zfill(25)
    board_1 = np.array([-int(bit) for bit in binary_string_1])
    board_2 = np.array([int(bit) for bit in binary_string_2])
    return board_1 + board_2

class ReinforcedMinMaxPlayer(Player) :
    def __init__(self, k : int = 0.1, epsilon : float = 0.1, alpha : float = 0.1, gamma : float = 0.9, depth : int = 1, opening_depth : int = 3):
        super().__init__()
        self.epsilon = epsilon # exploration rate
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.Q = dict()
        self.state = None
        self.action = None
        self.sequence = []
        self._used_moves = 0
        self.wins = 0
        self.learning = True
        self.power = 5
        self.games_played = 0
        self.depth = depth
        self.verbose = False
        self.opening_depth = opening_depth
        self.training = False

    def make_move(self, game : Game) -> tuple[tuple[int, int], Move]:
        # 1.1 Get the actual board
        self.state = game.get_board()
        # 1.2 Check if the player is O
        player_is_O = game.get_current_player() == 1
        # 2. Count the number of moves already made in the game
        already_made_moves = np.count_nonzero(self.state != -1) # number of moves already made by the player, obtained by counting the number of non-neutral pieces on the board
        

        if self.training:
            # 2.1 If we're still in the opening, use the opening book
            if already_made_moves < self.opening_depth: 
                # 3. If the player is O, transform the board to take the pov of player X
                if player_is_O:
                    for i in range(5):
                        for j in range(5):
                            self.state[i][j] = (1 - self.state[i][j]) if self.state[i][j] in [0, 1] else self.state[i][j]
                self.state = encode_board(self.state)
                # 3.1 If the state is not in the Q table, add it
                if self.state not in self.Q:
                    self.Q[self.state] = {action : random.uniform(0,1) for action in GameWrapper(game).get_possible_actions()}
                    self.action = max(self.Q[self.state], key= lambda k : self.Q[self.state][k])
                # 3.2 If the state is in the Q table, use the Q table
                else:
                    if random.uniform(0, 1) < self.epsilon:
                        self.action = random.choice(list(self.Q[self.state].keys()))
                    else:
                        self.action = max(self.Q[self.state], key= lambda k : self.Q[self.state][k])
                        self._used_moves += 1
                # 3.3 We're learning, add the state and action to the sequence
                self.sequence.append((self.state, self.action))
            # To better explore the opening, we use a random approach rather than minmax
            else:
                    self.action = {action : random.uniform(0,1) for action in GameWrapper(game).get_possible_actions()}
                    self.action = max(self.action, key= lambda k : self.action[k])

        else:
            # 2.1 If we're still in the opening, use the opening book
            if already_made_moves < self.opening_depth: 
                # 3. If the player is O, transform the board to take the pov of player X
                if player_is_O:
                    for i in range(5):
                        for j in range(5):
                            self.state[i][j] = (1 - self.state[i][j]) if self.state[i][j] in [0, 1] else self.state[i][j]
                self.state = encode_board(self.state)
                # 3.1 If the state is not in the Q table, take a random action
                if self.state not in self.Q:
                    self.action = {action : random.uniform(0,1) for action in GameWrapper(game).get_possible_actions()}
                    self.action = max(self.action, key= lambda k : self.action[k])
                # 3.2 If the state is in the Q table, use the Q table
                else:
                    self.action = max(self.Q[self.state], key= lambda k : self.Q[self.state][k])
                    self._used_moves += 1
            # 4. If we're not in the opening, use minimax
            else:
                minimax_player_id = game.get_current_player()
                maximizing_player = True

                # 2. Call minimax recursively for the current player
                _, best_action = self.minimax(game, self.depth, maximizing_player, minimax_player_id)
                self.action = best_action

        return self.action
    
    def update(self, reward : int) -> None: # reward is 1 if the player won, -1 if the player lost, 0.1 if draw
        for state, action in reversed(self.sequence):
            if reward == 1:
                self.Q[state][action] = min(1 , self.Q[state][action] + self.alpha * (1 * reward * self.Q[state][action]))
            else:
                self.Q[state][action] = max(0 , self.Q[state][action] + self.alpha * (1 * reward * self.Q[state][action]))
            reward = reward * self.gamma
        self.sequence = []
        self.games_played += 1

    class RandomPlayer(Player):
        def __init__(self) -> None:
            super().__init__()

        def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            return from_pos, move

    def stop_learning(self):
        self.learning = False
        self.epsilon = 0

    @property
    def used_moves(self):
        return self._used_moves
    
    def save(self,path : str):
        path = os.path.join(path, "player_reinforced_minmax.npy")
        np.save(path, self)
        return path
    
    def load(self,path : str):
        temp = np.load(path, allow_pickle=True).item()
        self.Q = temp.Q
        self.epsilon = temp.epsilon
        self.alpha = temp.alpha
        self.gamma = temp.gamma
        self.power = temp.power
        self.games_played = temp.games_played
        self._used_moves = temp._used_moves
        self.opening_depth = temp.opening_depth

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

    def train_against(self, opponent : Player, games : int = 1000, verbose : bool = False, plot : bool = False) -> None:
        self.training = True
        init_alpha = self.alpha
        init_epsilon = self.epsilon
        wins = [0,0]
        perc = []
        for i in range(games):
            g = Game()
            if i % 2 == 0:
                winner = g.play(self, opponent)
                wins[winner] += 1
                if winner == 0:
                    self.wins += 1
                    self.update(1)
                else:
                    self.update(-1)
            else:
                winner = g.play(opponent, self)
                wins[1-winner] += 1
                if winner == 1:
                    self.wins += 1
                    self.update(1)
                else:
                    self.update(-1)
            if (i + 1) % 50  == 0:
                if verbose:
                    print(f"{self.used_moves} already tried moves used with {self.Q.__len__()} states with epsilon {self.epsilon}")
                    print("Wins : ", wins[0] / (sum(wins)) * 100 , "%")
                if plot:
                    perc.append((wins[0] / (sum(wins)) , self.alpha, self.epsilon ))
                
                self.epsilon = init_epsilon*(1 - i/games)**0.9
                self.alpha = init_alpha*(1 - i/games)**0.9
                
            elif (i + 1) % 100  == 0 and plot:
                perc.append((wins[0] / (sum(wins)) , self.alpha, self.epsilon ))

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(perc)
            plt.show()
        if verbose:
            print(f"{self.used_moves} already tried moves used with {self.Q.__len__()} states with epsilon {self.epsilon}")
            print("Wins : ", wins[0] / (sum(wins)) * 100 , "%")
        self.training = False