from collections import namedtuple
from datetime import time
from math import ceil
import numpy as np
from random import random
from queue import PriorityQueue, SimpleQueue , LifoQueue
from functools import reduce
from enum import Enum
PROBLEM_SIZE = 5
NUM_SETS = 15

State = namedtuple('State', ['taken', 'untaken'])

class Algorithm(Enum):
    A_STAR = 1
    GREEDY = 2
    DEPTH_FIRST = 3
    BREADTH_FIRST = 4

class SetCoveringProblem:
    
    def __init__(self,algorithm : Algorithm) -> None:
        match algorithm:
            case Algorithm.A_STAR:
                self.frontier = PriorityQueue()
                self.cost_function = self.euristic
            case Algorithm.GREEDY:
                self.frontier = PriorityQueue()
                self.cost_function = self.distance
            case Algorithm.DEPTH_FIRST:
                self.frontier = LifoQueue() #it would be more correct to use deque() but it does not allow you to use the put method
                self.cost_function = lambda state : 0
            case Algorithm.BREADTH_FIRST:
                self.frontier = SimpleQueue()
                self.cost_function = lambda state : 0
        
    def goal_check(self,state):
        return np.all(self.covered(state))

    def covered(self,state):
        return reduce(np.logical_or,[self.sets[i] for i in state.taken],  np.array([False for _ in range(self.size)]) )

    def distance(self,state):
        return self.size - sum (
            reduce(np.logical_or,[self.sets[i] for i in state.taken], np.array([False for _ in range(self.size)]) ))

    def euristic(self,state):
        already_covered = self.covered(state)
        if np.all(already_covered):
            return 0
        largest_remaining_set_size = max(sum(np.logical_and(s, np.logical_not(already_covered))) for s in self.sets if not np.all(np.logical_and(s, state.taken)))
        missing_size = self.size - sum(already_covered)
        optimistic_estimate = ceil(missing_size / largest_remaining_set_size)
        return optimistic_estimate

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
            print("Solution found")
            print(f"Found solution in {format(self.counter)} iterations,  steps : ({len(self.current_state.taken)})")
            print(f"Solution : {self.current_state.taken}")
        else:
            print("No solution found")

if __name__ == "__main__":
    
    SETS = tuple([np.array([random() < .1 for _ in range(PROBLEM_SIZE)]) for _ in range(NUM_SETS)])
    while (SetCoveringProblem(Algorithm.A_STAR).check_solvable(SETS) == False):
        SETS = tuple([np.array([random() < .1 for _ in range(PROBLEM_SIZE)]) for _ in range(NUM_SETS)])

    print("A*")
    SetCoveringProblem(Algorithm.A_STAR).solve(SETS)

    print("greedy")
    SetCoveringProblem(Algorithm.GREEDY).solve(SETS)

    print("depth first")
    SetCoveringProblem(Algorithm.DEPTH_FIRST).solve(SETS)

    print("breadth first")
    SetCoveringProblem(Algorithm.BREADTH_FIRST).solve(SETS)
