Copyright **`(c)`** 2023 Alessandro Chiabodo `<s309234@studenti.polito.it>`  
[AChiabodo/compIntelligence](https://github.com/AChiabodo/compIntelligence.git)   

Project made in collaboration with Beatrice Occhiena S314971

## Introduction
This project is a simple implementation of various players for the game Quixo.
The main players that are implemented are:
- Simulative player: it uses a Monte Carlo-like approach where it simulates N games from the current state and chooses the move that leads to the best average result.
- Minimax player: it uses the minimax algorithm with alpha-beta pruning to choose the best move.
- Minimax player with opening tables: it uses a reinforcement learning approach to learn the best moves in the first N moves of the game and then uses the minimax algorithm with alpha-beta pruning to choose the best move.
- Reinforcement learning player: it uses a reinforcement learning approach to learn the best moves for the game.
Sadly I didn't have enough time to implement also the symmetry tables (made by Beatrice) in the players so the players are not as strong as they could be.
## Results
All players were tested against a Random player in N games where each time the starting player was chosen randomly.
- Reinforcement learning player: Required too many moves to learn, never got to a good level. Best result was 0.6% of wins.
- Simulative Player: Won nearly 100% of the games but required too much time for each move (almost two seconds per game).
- Minimax player: Won nearly 100% of the games and required a reasonable amount of time for each move (around 0.1 seconds per game) with a (very) low depth. Incresing the depth to 3 or 4 made the player too slow to be playable.
- Minimax player with opening tables: Won nearly 100% of the games and required a reasonable amount of time for each move (around 0.1 seconds per game) with a (very) low depth. Incresing the depth to 3 or 4 were almost useless because the player already knew the best moves for the first 3 or 4 moves of the game. The training time was reasonable (around 5 minutes for a good player).
## Choosen player
The choosen player is the Minimax player with opening tables because it is the fastest and the most reliable player.
To use this player you have to load the trained tables contained in the file `player_reinforced_minmax.npy` with the player.load function
The file main.py contains an example of how to use each player.