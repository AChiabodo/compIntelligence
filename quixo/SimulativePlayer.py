from copy import deepcopy
import numpy as np
import random

from GameWrapper import GameWrapper
from game import Game, Move, Player

class SimulativePlayer(Player) :
    def __init__(self):
        super().__init__()
        
        self.action = None

    def make_move(self, game : Game) -> tuple[tuple[int, int], Move]:
        self.current_player = game.get_current_player()
        self.action = self.find_best_move(game)
        
        return self.action
    
    def find_best_move(self, game : Game) -> tuple[tuple[int, int], Move]:
        best_moves = {}
        for from_pos,move in GameWrapper(game).get_possible_actions():
            g = GameWrapper(game)
            if g.make_move(from_pos, move, g.current_player_idx):
                tmp = self.simulate(g)
                best_moves[(from_pos, move)] = tmp
        return max(best_moves,key = lambda k : best_moves[k])
    
    def simulate(self, game: 'Game') -> int:
        size = 10
        winner = 0
        for _ in range(size):
            g = deepcopy(game)
            player1 = self.RandomPlayer()
            player2 = self.RandomPlayer()
            winner += (g.play(player1, player2) == self.current_player)
        return winner / size

    class RandomPlayer(Player):
        def __init__(self) -> None:
            super().__init__()

        def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            return from_pos, move