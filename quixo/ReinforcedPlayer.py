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

class ReinforcedPlayer(Player) :
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

        # 3. If the player is O, transform the board to take the pov of player X
        if player_is_O:
            for i in range(5):
                for j in range(5):
                    self.state[i][j] = (1 - self.state[i][j]) if self.state[i][j] in [0, 1] else self.state[i][j]
        self.state = encode_board(self.state)
        #self.state = tuple(map(tuple, self.state))
        if self.state not in self.Q:
            self.Q[self.state] = {action : random.uniform(0,1) for action in GameWrapper(game).get_possible_actions()}
            
            self.action = max(self.Q[self.state], key= lambda k : self.Q[self.state][k])
        else:
            if random.uniform(0, 1) < self.epsilon:
                self.action = random.choice(list(self.Q[self.state].keys()))
            else:
                self.action = max(self.Q[self.state], key= lambda k : self.Q[self.state][k])
                self._used_moves += 1
        if self.learning:
            self.sequence.append((self.state, self.action))
        return self.action
    
    def update(self, reward : int) -> None: # reward is 1 if the player won, -1 if the player lost, 0.1 if draw
        for state, action in reversed(self.sequence):
            if reward == 1:
                self.Q[state][action] = min(1 , self.Q[state][action] + self.alpha * ( 4 * reward * self.Q[state][action]))
            else:
                self.Q[state][action] = max(0 , self.Q[state][action] + self.alpha * (reward * self.Q[state][action]))
            reward = reward * self.gamma
        self.sequence = []
        self.games_played += 1
    
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

    def train_against(self, opponent : Player, games : int = 1000, verbose : bool = False, plot : bool = False) -> None:
        wins = [0,0]
        perc = []
        init_epsilon = self.epsilon
        init_alpha = self.alpha
        self.learning = True
        temp_wins = [0,0]
        for i in range(games):
            g = Game()
            if i % 2 == 0:
                winner = g.play(self, opponent)
                wins[winner] += 1
                if winner == 0:
                    self.wins += 1
                    self.update(reward=1)
                else:
                    self.update(reward=-1)
            else:
                winner = g.play(opponent, self)
                wins[1-winner] += 1
                if winner == 1:
                    self.wins += 1
                    self.update(1)
                else:
                    self.update(-1)
            if (i + 1) % 100  == 0:
                temp_wins[0] = wins[0] - temp_wins[0]
                temp_wins[1] = wins[1] - temp_wins[1]
                if verbose:
                    print(f"{self.used_moves} already tried moves used with {self.Q.__len__()} states with epsilon {self.epsilon}")
                    print("Wins : ", temp_wins[0] / (sum(temp_wins)) * 100 , "%")
                if plot:
                    perc.append((temp_wins[0] / (sum(temp_wins)) , self.alpha, self.epsilon ))
                self.epsilon *= 0.98
                self.alpha *= 0.97
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(perc)
            plt.show()
        self.learning = False
        self.epsilon = init_epsilon
        self.alpha = init_alpha