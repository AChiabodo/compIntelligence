
import random
import time
from MinMaxPlayer import MinmaxPlayer
from game import Game, Move, Player

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class MyPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

GAME_NUMBER = 50
if __name__ == '__main__':
    wins = [0, 0]
    player1 = MinmaxPlayer(depth=1)
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
    