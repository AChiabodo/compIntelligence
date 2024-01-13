
import random
import time
from SimulativePlayer import SimulativePlayer
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
    player1 = SimulativePlayer(power=5)
    player2 = RandomPlayer()
    current_time = time.time()
    for _ in range(GAME_NUMBER):
        print(f"At game {_} we are {wins}")
        g = Game()
        winner = g.play(player1, player2)
        wins[winner] += 1
    print(f"Final score: {wins}")
    print(f"Time elapsed: {time.time() - current_time}")
    print(f"Time per game: {(time.time() - current_time) / GAME_NUMBER}")