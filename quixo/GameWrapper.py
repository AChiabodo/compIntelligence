from game import Game, Move
from copy import deepcopy
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
