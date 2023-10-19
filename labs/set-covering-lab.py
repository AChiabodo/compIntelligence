from collections import namedtuple
import numpy as np
from random import random
from queue import PriorityQueue, SimpleQueue , LifoQueue
from functools import reduce

PROBLEM_SIZE = 10
NUM_SETS = 16

State = namedtuple('State', ['taken', 'untaken'])

def goal_check(state):
    return np.all(reduce(np.logical_or,[SETS[i] for i in state.taken],  np.array([False for _ in range(PROBLEM_SIZE)]) ))

def distance(state):
    return PROBLEM_SIZE - sum (
        reduce(np.logical_or,[SETS[i] for i in state.taken], np.array([False for _ in range(PROBLEM_SIZE)]) ))

def euristic(state):
    return distance(state) + len(state.taken)

SETS = tuple([np.array([random() < .2 for _ in range(PROBLEM_SIZE)]) for _ in range(NUM_SETS)])

while not goal_check(State(set(range(NUM_SETS)),set())) :
    print("No solution found, generating new problem")
    SETS = tuple([np.array([random() < .2 for _ in range(PROBLEM_SIZE)]) for _ in range(NUM_SETS)])    

frontier = PriorityQueue()
state = State(set(),set(range(NUM_SETS)))
frontier.put((euristic(state),state))
_ , current_state = frontier.get()
counter = 0
while not goal_check(current_state):
    counter += 1
    for action in current_state.untaken:
        new_state = State(current_state.taken.union({action}),current_state.untaken.difference({action}))
        frontier.put((euristic(new_state),new_state))
    _ , current_state = frontier.get()
assert goal_check(current_state), "Goal check failed"
print("Solution found with A*")
print(f"Found solution in {format(counter)} iterations,  steps : ({len(current_state.taken)})")
print(f"Solution : {current_state.taken}")


frontier = PriorityQueue()
state = State(set(),set(range(NUM_SETS)))
frontier.put((euristic(state),state))
_ , current_state = frontier.get()
counter = 0
while not goal_check(current_state):
    counter += 1
    for action in current_state.untaken:
        new_state = State(current_state.taken.union({action}),current_state.untaken.difference({action}))
        frontier.put((euristic(new_state),new_state))
    _ , current_state = frontier.get()
assert goal_check(current_state), "Goal check failed"
print("Solution found with greedy")
print(f"Found solution in {format(counter)} iterations,  steps : ({len(current_state.taken)})")
print(f"Solution : {current_state.taken}")


frontier = LifoQueue()
state = State(set(),set(range(NUM_SETS)))
frontier.put(state)
current_state = frontier.get()
counter = 0
while not goal_check(current_state):
    counter += 1
    for action in current_state.untaken:
        new_state = State(current_state.taken.union({action}),current_state.untaken.difference({action}))
        frontier.put(new_state)
    current_state = frontier.get()
assert goal_check(current_state), "Goal check failed"
print("Solution found with dept first")
print(f"Found solution in {format(counter)} iterations,  steps : ({len(current_state.taken)})")
print(f"Solution : {current_state.taken}")


frontier = SimpleQueue()
state = State(set(),set(range(NUM_SETS)))
frontier.put(state)
current_state = frontier.get()
counter = 0
while not goal_check(current_state):
    counter += 1
    for action in current_state.untaken:
        new_state = State(current_state.taken.union({action}),current_state.untaken.difference({action}))
        frontier.put(new_state)
    current_state = frontier.get()
assert goal_check(current_state), "Goal check failed"
print("Solution found with breadth first")
print(f"Found solution in {format(counter)} iterations,  steps : ({len(current_state.taken)})")
print(f"Solution : {current_state.taken}")