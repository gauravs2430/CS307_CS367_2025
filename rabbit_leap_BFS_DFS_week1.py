from collections import deque

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

def is_goal(state):
    return state == "WWW_EEE"

def get_successors(state):
    successors = []
    index = state.index('_')
    state_len = len(state)
    left = index - 1
    leftleft = index - 2
    right = index + 1
    rightright = index + 2
    
    # Move from left (E can move right)
    if left >= 0 and state[left] == 'E':
        newstate = list(state)
        newstate[left], newstate[index] = newstate[index], newstate[left]
        successors.append(''.join(newstate))
    elif leftleft >= 0 and state[leftleft] == 'E':
        newstate = list(state)
        newstate[leftleft], newstate[index] = newstate[index], newstate[leftleft]
        successors.append(''.join(newstate))
    
    # Move from right (W can move left)
    if right < state_len and state[right] == 'W':
        newstate = list(state)
        newstate[right], newstate[index] = newstate[index], newstate[right]
        successors.append(''.join(newstate))
    elif rightright < state_len and state[rightright] == 'W':
        newstate = list(state)
        newstate[rightright], newstate[index] = newstate[index], newstate[rightright]
        successors.append(''.join(newstate))
    
    return successors

# Reconstruct solution path from goal node to initial node
def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]

# BFS
def bfs():
    init_state = "EEE_WWW"
    goal_state = "WWW_EEE"
    visited = set()
    queue = deque()
    root = Node(init_state)
    queue.append(root)
    visited.add(init_state)
    
    while queue:
        curr_node = queue.popleft()
        if is_goal(curr_node.state):
            return reconstruct_path(curr_node)
        
        for succ in get_successors(curr_node.state):
            if succ not in visited:
                visited.add(succ)
                queue.append(Node(succ, curr_node))
    
    return None

# DFS Implementation
def dfs():
    initial_state = "EEE_WWW"
    goal_state = "WWW_EEE"
    visited = set()
    stack = []
    root = Node(initial_state)
    stack.append(root)
    visited.add(initial_state)
    
    while stack:
        current_node = stack.pop()
        if is_goal(current_node.state):
            return reconstruct_path(current_node)
        
        for succ in get_successors(current_node.state):
            if succ not in visited:
                visited.add(succ)
                stack.append(Node(succ, current_node))
    
    return None

# Run BFS
solution_bfs = bfs()
print("BFS Solution:")
for step_num, config in enumerate(solution_bfs):
    print(f"Step {step_num}: {config}")
print(f"\nTotal Steps Taken (BFS): {len(solution_bfs) - 1}")

# Run DFS
solution_dfs = dfs()
print("\nDFS Solution:")
for step_num, config in enumerate(solution_dfs):
    print(f"Step {step_num}: {config}")
print(f"\nTotal Steps Taken (DFS): {len(solution_dfs) - 1}")
