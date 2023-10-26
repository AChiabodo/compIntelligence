from collections import namedtuple
from datetime import time
from math import ceil
import numpy as np
from random import random , shuffle
from queue import PriorityQueue, SimpleQueue , LifoQueue
from functools import reduce
from enum import Enum
PROBLEM_SIZE = 20
NUM_SETS = 25

State = namedtuple('State', ['taken', 'untaken'])
PROBABILITY = 0.43

class SetCoveringProblem:
    """
    A class representing a set covering problem that can be solved by using the specified algorithm.

    Includes also the definition of the problem's state, the goal check function and the possibility of check that a problem is solvable or not.

    Attributes:
    - algorithm (Algorithm): the algorithm to use for solving the problem
    - MODE (Mode): the mode to use for solving the problem (either ACTION or DEBUG)

    Methods:
    - check_solvable(self, sets): checks if the given sets represents a solvable problem
    - solve(self, sets): solves the set covering problem over the given sets using the specified algorithm
    """
class SetCoveringProblem:
    
    class Algorithm(Enum):
        A_STAR = 1
        GREEDY = 2
        DEPTH_FIRST = 3
        BREADTH_FIRST = 4
        DIJKSTRA = 5

    class Mode(Enum):
        ACTION = 1
        DEBUG = 2

    def __init__(self,algorithm : Algorithm,MODE : Mode = Mode.DEBUG) -> None:
        self.mode = MODE
        match algorithm:
            case self.Algorithm.A_STAR:
                self.frontier = PriorityQueue()
                self.cost_function = self.euristic
            case self.Algorithm.GREEDY:
                self.frontier = PriorityQueue()
                self.cost_function = self.distance
            case self.Algorithm.BREADTH_FIRST:
                self.frontier = SimpleQueue()
                self.cost_function = lambda state : 0
            case self.Algorithm.DIJKSTRA:
                self.frontier = PriorityQueue()
                self.cost_function = lambda state : state.taken.__len__()
        
    def goal_check(self,state):
        return np.all(self.covered(state))

    def covered(self,state):
        return reduce(np.logical_or,[self.sets[i] for i in state.taken],  np.array([False for _ in range(self.size)]) )

    def distance(self,state):
        return self.size - sum (reduce(np.logical_or,[self.sets[i] for i in state.taken], np.array([False for _ in range(self.size)]) ))

    def euristic(self,state):
        already_covered = self.covered(state)
        if np.all(already_covered):
            return 0
        optimistic_estimate = 0
        # Sort sets by number of elements not already covered
        untaken_sets = sorted((sum(np.logical_and(s, np.logical_not(already_covered))) for s in [self.sets[i] for i in state.untaken]), reverse=True)

        # Add sets until all elements are covered
        for set in untaken_sets:
            missing_size = self.size - sum(already_covered)
            if missing_size <= 0:
                break
            if not np.all(np.logical_and(set, already_covered)):
                optimistic_estimate += 1
                already_covered = np.logical_or(already_covered, set)  
        return optimistic_estimate + state.taken.__len__()

    def check_solvable(self,sets):
        self.sets = sets
        self.size = len(self.sets[0])
        num_sets = len(sets)
        if not self.goal_check(State(set(range(num_sets)),set())): 
            return False
        else:
            return True
        
    def solve(self,sets):
        self.sets = sets
        self.size = len(self.sets[0])
        num_sets = len(sets)
        self.initial_state = State(set(),set(range(num_sets)))
        self.counter = 0  
        if not self.check_solvable(self.sets): raise Exception("Sets don't cover the problem")
        
        self.frontier.put((self.cost_function(self.initial_state),self.initial_state))
        
        _ , self.current_state = self.frontier.get()        
        while not self.goal_check(self.current_state):
            self.counter += 1
            for action in self.current_state.untaken:
                self.new_state = State(self.current_state.taken.union({action}),self.current_state.untaken.difference({action}))
                self.frontier.put((self.cost_function(self.new_state),self.new_state))
            _ , self.current_state = self.frontier.get()
        if(self.goal_check(self.current_state)):
            if self.mode == self.Mode.DEBUG:
                print("Solution found")
                print(f"Found solution in {format(self.counter)} iterations,  steps : ({len(self.current_state.taken)})")
                print(f"Solution : {self.current_state.taken}")
            return self.current_state.taken
        else:
            return None


if __name__ == "__main__":
    compare = SetCoveringProblem.Algorithm.GREEDY
    MODE = SetCoveringProblem.Mode.ACTION
    if PROBABILITY < 0 or PROBABILITY > 1:
        raise Exception("Probability must be between 0 and 1")

### TESTING A* ALGORITHM OPTIMALITY ###
    for i in range(100):    

        SETS = tuple([np.array([random() < PROBABILITY for _ in range(PROBLEM_SIZE)]) for _ in range(NUM_SETS)])
        while (SetCoveringProblem(SetCoveringProblem.Algorithm.A_STAR).check_solvable(SETS) == False):
            SETS = tuple([np.array([random() < PROBABILITY for _ in range(PROBLEM_SIZE)]) for _ in range(NUM_SETS)])
        print("iteration : ",i)

        stateA = SetCoveringProblem(SetCoveringProblem.Algorithm.A_STAR,MODE=MODE).solve(SETS)
        match compare:
            case SetCoveringProblem.Algorithm.GREEDY:
                stateB = SetCoveringProblem(SetCoveringProblem.Algorithm.GREEDY,MODE=MODE).solve(SETS)

            case SetCoveringProblem.Algorithm.DIJKSTRA:
                stateB = SetCoveringProblem(SetCoveringProblem.Algorithm.DIJKSTRA,MODE=MODE).solve(SETS)

            case SetCoveringProblem.Algorithm.BREADTH_FIRST:
                stateB = SetCoveringProblem(SetCoveringProblem.Algorithm.BREADTH_FIRST,MODE=MODE).solve(SETS)

        if stateA.__len__() > stateB.__len__():
            print("ERROR")
            print(SETS)
            print(stateA)
            print(stateB)
            break

### A* ALGORITHM ORDER INDEPENDENCE TEST ###
    SETS = tuple([np.array([random() < PROBABILITY for _ in range(PROBLEM_SIZE)]) for _ in range(NUM_SETS)])
    while (SetCoveringProblem(SetCoveringProblem.Algorithm.A_STAR).check_solvable(SETS) == False):
        SETS = tuple([np.array([random() < PROBABILITY for _ in range(PROBLEM_SIZE)]) for _ in range(NUM_SETS)])
    
    stateA = SetCoveringProblem(SetCoveringProblem.Algorithm.A_STAR,MODE=MODE).solve(SETS)
    for i in range(10):
        print("iteration : ",i)
        lista = list(SETS)
        shuffle(lista)
        stateB = SetCoveringProblem(SetCoveringProblem.Algorithm.A_STAR,MODE=MODE).solve(tuple(lista))
        if stateA.__len__() > stateB.__len__():
            print("ERROR")
            print(SETS)
            print(stateA)
            print(stateB)
            break