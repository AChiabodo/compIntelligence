"""
    symmetry.py

    #TODO: Add a description of the file
    Rotations in numpy are COUNTER-CLOCKWISE:
    - https://numpy.org/doc/stable/reference/generated/numpy.rot90.html
"""

import numpy as np
from game import Move

class SymmetryGenerator:

    def __init__(self):
        self.board_transforms = [
            ("identity", self.board_identity),
            ("rotate_90", self.board_rotate_90),
            ("rotate_180", self.board_rotate_180),
            ("rotate_270", self.board_rotate_270),
            ("reflect_horizontal", self.board_reflect_horizontal),
            ("reflect_vertical", self.board_reflect_vertical),
            ("reflect_diagonal", self.board_reflect_diagonal),
            ("reflect_antidiagonal", self.board_reflect_antidiagonal)
        ]
        self.coor_transforms = [
            ("identity", self.coor_identity),
            ("rotate_90", self.coor_rotate_90),
            ("rotate_180", self.coor_rotate_180),
            ("rotate_270", self.coor_rotate_270),
            ("reflect_horizontal", self.coor_reflect_horizontal),
            ("reflect_vertical", self.coor_reflect_vertical),
            ("reflect_diagonal", self.coor_reflect_diagonal),
            ("reflect_antidiagonal", self.coor_reflect_antidiagonal)
        ]
        self.slide_transforms = [
            ("identity", self.slide_identity),
            ("rotate_90", self.slide_rotate_90),
            ("rotate_180", self.slide_rotate_180),
            ("rotate_270", self.slide_rotate_270),
            ("reflect_horizontal", self.slide_reflect_horizontal),
            ("reflect_vertical", self.slide_reflect_vertical),
            ("reflect_diagonal", self.slide_reflect_diagonal),
            ("reflect_antidiagonal", self.slide_reflect_antidiagonal)
        ]

    ############################
    # SYMMETRIES FOR THE BOARD #
    ############################

    def board_identity(self, board):
        return board

    def board_rotate_90(self, board: np.ndarray):
        return np.rot90(board)

    def board_rotate_180(self, board: np.ndarray):
        return np.rot90(board, k=2)

    def board_rotate_270(self, board: np.ndarray):
        return np.rot90(board, k=3)

    def board_reflect_horizontal(self, board: np.ndarray):
        return np.fliplr(board)

    def board_reflect_vertical(self, board: np.ndarray):
        return np.flipud(board)

    def board_reflect_diagonal(self, board: np.ndarray):
        return np.transpose(board)

    def board_reflect_antidiagonal(self, board: np.ndarray):
        return np.flip(np.transpose(board), axis=1)

    def board_get_symmetries(self, board: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """
            Returns a list of all the symmetries of the given board.
            - Each symmetry is a tuple containing:
                1. the label of the transformation
                2. the transformed board
        """
        symmetries = [(label, transform(board)) for label, transform in self.board_transforms]
        return symmetries
    
    def get_base_state(self, board) -> (str, np.ndarray):
        """
            Maps a state to its `base state`.
            - The base state is the state with the lowest lexicographical order among all the states that are equivalent to the given state under the symmetries of the board.
            - OB: get the base state to be saved in the Q-table
        """

        # 1. Get the symmetries of the board
        all_symmetries = self.board_get_symmetries(board)

        # 2. Find the lexicographically minimum state and its transformation label
        transf_performed, base_state = min(all_symmetries, key=lambda x: x[1].tobytes())

        # 3. Return the minimum state and its transformation label
        return transf_performed, base_state
    
    ##########################
    # SYMMETRIES FOR ACTIONS #
    ##########################

    def get_base_action(self, from_pos: tuple[int, int], slide: Move, symmetry_to_apply: str) -> (tuple[int, int], Move):
        """
            Maps an action to the base action that is transformed by the given symmetry.
            - OB: get the action to save in the Q-table with the base state
        
        """
        # 1. Get the transformations to apply
        coor_transf = [transform for label, transform in self.coor_transforms if label == symmetry_to_apply][0]
        slide_transf = [transform for label, transform in self.slide_transforms if label == symmetry_to_apply][0]

        # 2. Apply the symmetry to the from_pos
        from_pos = coor_transf(from_pos)

        # 3. Apply the symmetry to the slide
        slide = slide_transf(slide)

        # 4. Return the transformed action
        return from_pos, slide
    
    def get_original_action(self, from_pos: tuple[int, int], slide: Move, symmetry_to_reverse: str) -> (tuple[int, int], Move):
        """
            Maps a base action to its original action reversing the transformation it was subjected to.
            - OB: get the action to use in the game
        """
        # 1. Get the reverse transformation
        if symmetry_to_reverse == "rotate_90":
            symmetry_to_apply = "rotate_270"
        elif symmetry_to_reverse == "rotate_180":
            symmetry_to_apply = "rotate_180"
        elif symmetry_to_reverse == "rotate_270":
            symmetry_to_apply = "rotate_90"
        else:
            symmetry_to_apply = symmetry_to_reverse
        
        # 2. Get the transformation to reverse
        coor_transf = [transform for label, transform in self.coor_transforms if label == symmetry_to_apply][0]
        slide_transf = [transform for label, transform in self.slide_transforms if label == symmetry_to_apply][0]

        # 3. Apply the transformation to the from_pos
        from_pos = coor_transf(from_pos)

        # 4. Apply the transformation to the slide
        slide = slide_transf(slide)

        # 5. Return the original action
        return from_pos, slide

    # COORDINATES SYMMETRIES

    def coor_identity(self, from_pos: tuple[int, int]):
        # Return the position (y, x) unchanged
        return from_pos
    
    def coor_rotate_90(self, from_pos: tuple[int, int]):
        # Rotate a position (y, x) by 90 degrees on a 5x5 board
        return from_pos[1], 4-from_pos[0]
    
    def coor_rotate_180(self, from_pos: tuple[int, int]):
        # Rotate a position (y, x) by 180 degrees on a 5x5 board
        return 4-from_pos[0], 4-from_pos[1]
    
    def coor_rotate_270(self, from_pos: tuple[int, int]):
        # Rotate a position (y, x) by 270 degrees on a 5x5 board
        return 4-from_pos[1], from_pos[0]
    
    def coor_reflect_horizontal(self, from_pos: tuple[int, int]):
        # Reflect a position (y, x) horizontally on a 5x5 board
        return from_pos[0], 4-from_pos[1]

    def coor_reflect_vertical(self, from_pos: tuple[int, int]):
        # Reflect a position (y, x) vertically on a 5x5 board
        return 4-from_pos[0], from_pos[1]
    
    def coor_reflect_diagonal(self, from_pos: tuple[int, int]):
        # Reflect a position (y, x) along the diagonal on a 5x5 board
        return from_pos[1], from_pos[0]

    def coor_reflect_antidiagonal(self, from_pos: tuple[int, int]):
        # Reflect a position (y, x) along the antidiagonal on a 5x5 board
        return 4-from_pos[1], 4-from_pos[0]
    
    # SLIDE MOVE SYMMETRIES

    def slide_identity(self, slide: Move):
        # Return the slide unchanged
        return slide
    
    def slide_rotate_90(self, slide: Move):
        # Rotate a slide by 90 degrees
        if slide == Move.TOP: return Move.LEFT
        elif slide == Move.BOTTOM: return Move.RIGHT
        elif slide == Move.LEFT: return Move.BOTTOM
        elif slide == Move.RIGHT: return Move.TOP

    def slide_rotate_180(self, slide: Move):
        # Rotate a slide by 180 degrees
        if slide == Move.TOP: return Move.BOTTOM
        elif slide == Move.BOTTOM: return Move.TOP
        elif slide == Move.LEFT: return Move.RIGHT
        elif slide == Move.RIGHT: return Move.LEFT

    def slide_rotate_270(self, slide: Move):
        # Rotate a slide by 270 degrees
        if slide == Move.TOP: return Move.RIGHT
        elif slide == Move.BOTTOM: return Move.LEFT
        elif slide == Move.LEFT: return Move.TOP
        elif slide == Move.RIGHT: return Move.BOTTOM

    def slide_reflect_horizontal(self, slide: Move):
        # Reflect a slide horizontally
        if slide == Move.TOP: return Move.TOP
        elif slide == Move.BOTTOM: return Move.BOTTOM
        elif slide == Move.LEFT: return Move.RIGHT
        elif slide == Move.RIGHT: return Move.LEFT

    def slide_reflect_vertical(self, slide: Move):
        # Reflect a slide vertically
        if slide == Move.TOP: return Move.BOTTOM
        elif slide == Move.BOTTOM: return Move.TOP
        elif slide == Move.LEFT: return Move.LEFT
        elif slide == Move.RIGHT: return Move.RIGHT

    def slide_reflect_diagonal(self, slide: Move):
        # Reflect a slide along the diagonal
        if slide == Move.TOP: return Move.LEFT
        elif slide == Move.BOTTOM: return Move.RIGHT
        elif slide == Move.LEFT: return Move.TOP
        elif slide == Move.RIGHT: return Move.BOTTOM

    def slide_reflect_antidiagonal(self, slide: Move):
        # Reflect a slide along the antidiagonal
        if slide == Move.TOP: return Move.RIGHT
        elif slide == Move.BOTTOM: return Move.LEFT
        elif slide == Move.LEFT: return Move.BOTTOM
        elif slide == Move.RIGHT: return Move.TOP