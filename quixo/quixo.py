"""
    quixo.py

    #TODO: Add a description of the file
"""

from game import Game, Move, Player
from symmetry import SymmetryGenerator


class Quixo(Game):
    def __init__(self) -> None:
        super().__init__()
    
    def print(self, winner: int=-1):
        """
        Prints the current player and the board in a more readable way
        - ⬜ are neutral pieces
        - ❌ are pieces of player 0
        - 🔘 are pieces of player 1
        """

        # 1. Print the board
        print("\n*****************")
        for row in self._board:
            for cell in row:
                if cell == -1:
                    print("⬜", end=" ")
                elif cell == 0:
                    print("❌", end=" ")
                elif cell == 1:
                    print("🔘", end=" ")
            print()
        print()

        # 2. Print the current player or the winner
        if winner >= 0:
            symbol = "❌" if winner == 0 else "🔘"
            print(f"Player {symbol} wins the game!")
        else:
            symbol = "❌" if self.current_player_idx == 0 else "🔘"
            print(f"Current player: {symbol}")

    def get_possible_actions(self) -> list[tuple[tuple[int, int], Move]]:
        """
        Returns a list of possible actions.

        An action is a tuple of:
        - the position of the piece to move
        - the direction in which to move it

        An action is possible if:
        - the piece is neutral or belongs to the current player
        - the piece is on the edge of the board

        The direction in which a piece can be moved depends on its position:
        - all pieces can be moved from the opposite edge(s) of the board
        - pieces not in a corner can be also moved in directions parallel to the edge they were taken from
        """
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
    
    def make_move(self, from_pos: tuple[int, int], slide: Move) -> bool:
        """Makes a move on the board."""
        return super()._Game__move(from_pos, slide, self.current_player_idx)
    
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

    def check_winner(self) -> int:
        """ Checks if there is a winner."""
        return super().check_winner()

    def change_player(self):
        """Changes the current player."""
        self.current_player_idx = 1 - self.current_player_idx

    def play(self, player1: Player, player2: Player, verbose: bool=False, debug: bool=False) -> int:
        '''
          This function is a modified version of the play() function in the Game class.
          It allows the user to have more control over the visualization of each step of the game.
          - `verbose`: if True, prints the current player, the board and the moves tried by the players
          - `debug`: if True, prints the number and the list of all possible actions for the current player 
        '''

        # 1. Initialize the players
        players = [player1, player2]
            
        # 2. Play the game
        winner = -1
        while winner < 0:

            # 2.0. Change the current player
            self.change_player()

            # 2.1. Print the board if verbose
            if verbose:
                self.print()
            
            # 2.2. DEBUG: Print the possible actions
            if debug:
                possible_actions = self.get_possible_actions()
                print("Number of possible actions:", len(possible_actions))
                print("Possible actions: ", possible_actions)
                
            # 2.3. Make the move of the current player
            ok = False
            while not ok:
                
                # 2.2.1. Get the move from the current player
                from_pos, slide = players[self.current_player_idx].make_move(self)
                
                # 2.2.2. Make the move and check if it was successful
                ok = self.make_move(from_pos, slide)

                # 2.2.3. VERBOSE: Print the move's outcome
                if verbose:
                    if ok:
                      print(f"Player {self.current_player_idx} moved ({from_pos}, {slide})")
                    else:
                      print(f"Player {self.current_player_idx} tried to move ({from_pos}, {slide}) but failed")
            
            # 2.4. Check if there is a winner
            winner = self.check_winner()

        # 3. VERBOSE: Print the final board and the winner
        if verbose:
            self.print(winner)
        
        # 4. Return the winner
        return winner