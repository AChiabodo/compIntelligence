from game import Game, Move, Player
from copy import deepcopy
import numpy as np
import random
class GameWrapper(Game):
    def __init__(self, game : Game) -> None:
        super().__init__() #useless?
        self._board = game.get_board()
        self.current_player_idx = game.get_current_player()

    def move(self, from_pos: tuple[int, int], move: Move, player: int) -> None:
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


class SimulativePlayer(Player):
    def __init__(self, power : int) -> None:
        super().__init__()
        self.power = power
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        best = 0
        best_move = None
        already_tried = 0
        while already_tried < self.power:
            g = GameWrapper(game)
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            if g.move(from_pos, move, g.current_player_idx):
                already_tried += 1
                tmp = self.simulate(g)
                if tmp > best:
                    best = tmp
                    best_move = (from_pos, move)
        if best_move is None:
            return (random.randint(0, 4), random.randint(0, 4)), random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return best_move
    
    def simulate(self, game: 'Game') -> int:
        winner = 0
        for _ in range(10):
            g = deepcopy(game)
            player1 = self.RandomPlayer()
            player2 = self.RandomPlayer()
            winner += (g.play(player1, player2) == 0)
        return winner

    class RandomPlayer(Player):
        def __init__(self) -> None:
            super().__init__()

        def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            return from_pos, move