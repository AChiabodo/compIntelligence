**Author**: Alessandro Chiabodo
 - Student ID: `s309234`
 - Institutional Email: `s309234@studenti.polito.it`
 - Repository : [AChiabodo/compIntelligence](https://github.com/AChiabodo/compIntelligence.git)
 
Resources Used:
 - [squillero/computational-intelligence](https://github.com/squillero/computational-intelligence) : Main source of material for the lab requirements and code
 - Stuart Russel, Peter Norvig, *Artificial Intelligence: A Modern Approach* [4th edition] : Partial source for ES algorithms
 - [Wikipedia](https://en.wikipedia.org/wiki/Nim) : Main source for the game rules
 - [beatrice-occhiena/Computational_intelligence](https://github.com/beatrice-occhiena/Computational_intelligence.git) : Some explanations and examples of rule-based expert agents

# Lab2 - ES Algorithm for Nim
Nim is a mathematical strategy game in which two players take turns removing (or "nimming") objects from different piles or stacks. In this version of the game (called 'misère') in each turn, a player can remove one or more objects from a single row, and the player who removes the last object loses.

## Task 0: Definition of Classes and Utils functions
In addition to the NimSum function already provided and used both for the expert agent and in the 'benchmarking' phase of the algorithm, several new classes and functions were defined and subsequently used in both assigned tasks.
 - *Player class* : main interface for interoperability between different players classes, contains the definition of the main methods called "play", used to define the move of the player
 - *Multiple Matches* : util function that calculate the outcomes of N different games where each time the starting player is randomly choosen

## Task 1: Find a rule-based Nim Player
To develope an "expert" agent able to win almost all games we'll need to find some basic rules that'll allow our algorithm to choose the best move out of all possible
 - A first attempt can be seen in `EasyRuleBasedPlayer` that analizes only the number of non-void columns and act accordingly
 - Another example can be seen in the `OptimalPlayer` based on the analisys of the NimSum of each move (the code is based on the original implementation from [squillero/computational-intelligence](https://github.com/squillero/computational-intelligence) )  
 
 Neither has a high win rate when used against another player who is slightly more intelligent than a "random" player.  
 Now we've two alternatives: optimize the Rule-based agent or follow a *simulative* approach.  
My results for both alternatives are:
 - `AdvancedRuleBasedPlayer` : An optimized version of the original `OptimalPlayer` that also uses the number of "big" and "non-empty" rows to make its decisions, allowing him to have an 80% win rate against the previous expert agent
 - `SimulativePlayer` : Following a more computational-heavy path, here we're computing all possible outcomes given by each move and then choosing the one that allows an higher win probability
The last two models are almost equal in terms of win rate, with `SimulativePlayer` slightly better, but with higher computational cost and lower scalability.
Now that we've defined some agents that are able to win almost certainly a game is time to create a more "intelligent" player that can learn from its mistakes

## Task 2: Define an Evolution Strategy
Evolution strategies and, more generally, evolutionary algorithms allow us to "train" an intelligent player.  
To achieve that we need a way to define some "rules" that our new agent will try to learn and optimize, and a possible (but computationally intense) way of doing so is to define a rule for each possible state that we could encounter in the game.  
In other words, we calculate *all possible combinations* of rows in the game, and for each individual in our player population, we assign *one (random) move* to each of them.  
To obtain an individual (or player) able to compete with the expert agents defined before we'll need to *train* them, evaluate them and choose only the best at each generation.  
The *fitness* of each individual is given by the wins in a fixed number (100) of matches against the previously defined expert agents.  

To obtain something usefull we'll need at least 40 generations of individuals (or better 40 *epoch* given that the hyperparameters are modified for each generation) and a starting population of at least 30/40 individuals.

There are also two different training method implemented in the class, the first based on "modern" ES that uses both crossover and mutation, and another with only the mutation of the parent.

In the end the resulting individuals are able to play at the same level of the best expert-agent that we've been able di define before, is also interesting to see how different players end up with almost the same rule-set as the algorithm converge to an (almost) optimal state.

Example of win-rate of a fully trained player :  
win ratio against AdvancedRuleBasedPlayer :  `62`  
win ratio against Simulative Player :  `66`   
win ratio against Random Player :  `88`

Note that to add the *K-constrain* (max number of pieces per move) is required to retrain all players from scratch