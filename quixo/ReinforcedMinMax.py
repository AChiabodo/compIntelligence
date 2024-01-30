from math import e
from game import Game, Move, Player
from copy import deepcopy
import numpy as np
import random
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
    def __init__(self, k : int = 0.1, epsilon : float = 0.1, alpha : float = 0.1, gamma : float = 0.9):
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

    def make_move(self, game : Game) -> tuple[tuple[int, int], Move]:
        self.state = game.get_board()
                # 2. Check if the player is O
        player_is_O = game.get_current_player() == 1
        already_made_moves = 2 # number of moves already made by the player, obtained by counting the number of non-neutral pieces on the board
        if self.learning and already_made_moves < 2: 
            # 3. If the player is O, transform the board to take the pov of player X
            if player_is_O:
                for i in range(5):
                    for j in range(5):
                        self.state[i][j] = (1 - self.state[i][j]) if self.state[i][j] in [0, 1] else self.state[i][j]
            self.state = encode_board(self.state)

            if self.state not in self.Q:
                self.Q[self.state] = {action : random.uniform(0,1) for action in GameWrapper(game).get_possible_actions()}

                self.action = max(self.Q[self.state], key= lambda k : self.Q[self.state][k])
            else:
                if random.uniform(0, 1) < self.epsilon:
                    self.action = random.choice(list(self.Q[self.state].keys()))
                else:
                    self.action = max(self.Q[self.state], key= lambda k : self.Q[self.state][k])
                    self._used_moves += 1
            self.sequence.append((self.state, self.action))
        else:
            minimax_player_id = game.get_current_player()
            maximizing_player = True

            # 2. Call minimax recursively for the current player
            _, best_action = self.minimax(game, self.depth, maximizing_player, minimax_player_id)

        return self.action
    
    def update(self, reward : int) -> None: # reward is 1 if the player won, -1 if the player lost, 0.1 if draw
        for state, action in reversed(self.sequence):
            if self.sequence.__len__() == 3 and reward == 1:
                self.Q[state][action] = min(1 , self.Q[state][action] + self.alpha * (2 *reward * self.Q[state][action]))
            elif reward == 1:
                self.Q[state][action] = min(1 , self.Q[state][action] + self.alpha * (reward * self.Q[state][action]))
            else:
                self.Q[state][action] = max(0 , self.Q[state][action] + self.alpha * (2 * reward * self.Q[state][action]))
            reward = reward * self.gamma
        self.sequence = []
        self.games_played += 1

    def find_good_moves(self, game : Game) -> tuple[tuple[int, int], Move]:
        best_moves = {}
        already_tried = 0
        while already_tried < self.power:
            g = GameWrapper(game)
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            if g.move(from_pos, move, g.current_player_idx):
                already_tried += 1
                tmp = self.simulate(g)
                best_moves[(from_pos, move)] = tmp
        return best_moves
    
    def find_best_moves(self, game : Game) -> tuple[tuple[int, int], Move]:
        best_moves = {}
        for from_pos,move in GameWrapper(game).get_possible_actions():
            g = GameWrapper(game)
            if g.move(from_pos, move, g.current_player_idx):
                tmp = self.simulate(g)
                best_moves[(from_pos, move)] = tmp
        return best_moves
    
    def simulate(self, game: 'Game') -> int:
        size = 10
        winner = 0
        for _ in range(size):
            g = deepcopy(game)
            player1 = self.RandomPlayer()
            player2 = self.RandomPlayer()
            winner += (g.play(player1, player2) == 0)
        return winner / size

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
    
    def save(self):
        np.save("player.npy", self)
    
    def load(self):
        temp = np.load("player.npy", allow_pickle=True).item()
        self.Q = temp.Q
        self.epsilon = temp.epsilon
        self.alpha = temp.alpha
        self.gamma = temp.gamma
        self.power = temp.power
        self.games_played = temp.games_played

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
