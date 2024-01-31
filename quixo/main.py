
from enum import Enum
import random
import time
import os
from MinMaxPlayer import MinmaxPlayer
from ReinforcedMinMax import ReinforcedMinMaxPlayer
from ReinforcedPlayer import ReinforcedPlayer
from game import Game, Move, Player

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class PlayerEnum(Enum):
    MINMAX = 0
    REINFORCED = 1
    REINFORCED_MINMAX = 2

CHOSEN_PLAYER = PlayerEnum.REINFORCED_MINMAX
TRAINING_IF_REINFORCED = False
GAME_NUMBER = 200

if __name__ == '__main__':
    wins = [0, 0]
    match CHOSEN_PLAYER:
        case PlayerEnum.MINMAX:
            player1 = MinmaxPlayer(depth=1)
        case PlayerEnum.REINFORCED:
            player1 = ReinforcedPlayer()
            path = os.path.join(os.getcwd(), "quixo" , "player_reinforced.npy")
            if os.path.exists(path):
                player1.load(path=path)
            elif TRAINING_IF_REINFORCED:
                player1.train_against(RandomPlayer(), 10000,verbose=True,plot=False)
        case PlayerEnum.REINFORCED_MINMAX:
            player1 = ReinforcedMinMaxPlayer(depth=1, opening_depth=3, epsilon=0.8, alpha=0.2, gamma=0.7)
            path = os.path.join(os.getcwd(), "quixo" , "player_reinforced_minmax.npy")
            if os.path.exists(path):
                player1.load(path=path)
            elif TRAINING_IF_REINFORCED:
                player1.train_against(RandomPlayer(), 10000,verbose=True,plot=False)
    player2 = RandomPlayer()
    current_time = time.time()
    for _ in range(GAME_NUMBER):
        if _ % 1 == 0 and _ != 0:        
            print(f"At game {_} we are {wins[0]/(wins[0] + wins[1]) * 100}")
        g = Game()
        if _ % 2 == 0:
            winner = g.play(player1, player2)
            wins[winner] += 1
        else:
            winner = g.play(player2, player1)
            wins[1-winner] += 1

    print(f"Final score: {wins}")
    print(f"Time elapsed: {time.time() - current_time}")
    print(f"Time per game: {(time.time() - current_time) / GAME_NUMBER}")
    