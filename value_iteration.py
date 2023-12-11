import numpy as np
from itertools import product
from copy import copy

class GridWorldMDP:
    def __init__(self, objects, object_coords, receptacle_coords, nT, grid_size, gamma=0.9):
        self.grid_size = grid_size
        self.n_objects = len(objects)
        self.gamma = gamma
        self.actions = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']
        # self.receptacle_coords = [(15, 5), (5, 15), (8, 12), (12, 8)]
        self.receptacle_coords = receptacle_coords
        self.objects = objects
        self.object_coords = object_coords
        self.objects_vertices = self.get_vertices(self.object_coords)
        self.receptacle_vertices = self.get_vertices(self.receptacle_coords)
        self.vertex_coords = [v for vertex in self.objects_vertices for v in vertex]
        self.nT = nT # number of tosses
        self.nP = self.n_objects - self.nT # number of pick and places

        perm_nP = self.permute_nP(self.nP)
        perm_nP_copy = copy(perm_nP)
        for ele in perm_nP_copy:
            # cannot have more than one object being pick and placed
            if ele.count(0) > 1:
                perm_nP.remove(ele)
        self.out = [vT + vP for vT in self.permute_nT(self.nT) for vP in perm_nP]
        self.states = [(i, j, k) for i in range(grid_size) for j in range(grid_size) for k in self.out]
        self.values = np.zeros((grid_size, grid_size, len(self.out)))
        self.policy = np.zeros((grid_size, grid_size, len(self.out)), dtype=int)
        print(len(self.out))
        self.reached_recept = False

    # 1 means object is on floor, 0 means object has been tossed
    def permute_nT(self, n):
        permutations = []
        for vals in product([0, 1], repeat=n):
            ele = np.array(vals).reshape((n, ))
            permutations.append(list(ele))
        return permutations
    
    # 1 means object is on floor, 0 means object has been picked up, and -1 means object has been placed
    def permute_nP(self, n):
        permutations = []
        for vals in product([-1, 0, 1], repeat=n):
            ele = np.array(vals).reshape((n, ))
            permutations.append(list(ele))
        return permutations
    
    def main_reward(self, state, action, next_state):
        obj_nP_status = next_state[-1][self.nT:]
        try:
            idx = obj_nP_status.index(0)
        except:
            idx = None

        if self.reached_recept:
            self.reached_recept = False
            # 50 reward for placing object at receptacle
            return 50
        
        # next_state = (next_state[0], next_state[1])
        if idx is None:
            return self.usual_reward(state, action, next_state)
        else:
            return self.receptacle_reward(state, action, next_state)
        
    def usual_reward(self, state, action, next_state):
        status = next_state[-1]
        next_state = (next_state[0], next_state[1])
        if next_state in self.object_coords:
            # penalty for colliding with objects and receptacle
            idx = self.object_coords.index(next_state)
            if status[idx] == 1:
                return -1
            else:
                return -0.01
            # return -1
        elif next_state in self.receptacle_coords:
            return -1
        elif next_state in self.vertex_coords: 
            idx = self.vertex_coords.index(next_state) // 4
            primitive = self.objects[idx][2]
            if state[-1][idx] == 1 and primitive == 1:
                # 100 reward for tossing
                return 100
            elif state[-1][idx] == 1 and primitive == 0:
                # 50 reward for pick and place
                return 50
            else:
                return -0.01
        else: 
            return -0.01
        
    def receptacle_reward(self, state, action, next_state):
        status = next_state[-1]
        next_state = (next_state[0], next_state[1])
        if next_state in self.object_coords:
            # penalty for colliding with objects
            idx = self.object_coords.index(next_state)
            if status[idx] == 1:
                return -10
            else:
                return -0.01
            # return -10
        elif next_state in self.receptacle_coords:
            # penalty for colliding with receptacle
            return -1
        else:
            return -0.01

    def is_valid_state(self, state):
        i, j, _ = state
        return 0 <= i < self.grid_size and 0 <= j < self.grid_size

    def get_vertices(self, objects):
        vertices = []
        length = 2
        for obj in objects:
            v1 = (obj[0] - length/2, obj[1] - length/2)
            v2 = (obj[0] + length/2, obj[1] + length/2)
            v3 = (obj[0] - length/2, obj[1] + length/2)
            v4 = (obj[0] + length/2, obj[1] - length/2)
            vertices.append((v1, v2, v3, v4))
        return vertices

    def transition(self, state, action):
        i, j, rest = state
        
        directions = {'n': (0, 1), 
                      'ne': (1, 1), 
                      'e': (1, 0), 
                      'se': (1, -1), 
                      's': (0, -1), 
                      'sw': (-1, -1), 
                      'w': (-1, 0), 
                      'nw': (-1, 1)}
        
        if action in directions:
            di, dj = directions[action]
            next_state = (i + di, j + dj, copy(rest))
        else: 
            raise ValueError("Invalid action")
        
        obj_nP_status = state[-1][self.nT:]
        try:
            idx = obj_nP_status.index(0)
        except:
            idx = None

        if self.is_valid_state(next_state):
            i_n, j_n, _ = next_state
            if idx is None:
                # no object to be placed
                if (i_n, j_n) in self.vertex_coords:
                    idx = self.vertex_coords.index((i_n, j_n)) // 4
                    if next_state[-1][idx] == 1:
                        next_state[-1][idx] = 0               
            else:
                # object to be placed
                if (i_n, j_n) in self.receptacle_vertices[self.objects[self.nT + idx][-1]] and next_state[-1][self.nT + idx] == 0:
                    next_state[-1][self.nT + idx] = -1
                    self.reached_recept = True
            return next_state
        else:
            return state
              
    def value_iteration(self, num_iterations):
        for iteration in range(num_iterations):
            new_values = np.copy(self.values)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    for k in self.out:
                        max_value = float('-inf')
                        for action in self.actions:
                            next_state = self.transition((i, j, k), action)
                            reward = self.main_reward((i, j, k), action, next_state)
                            next_value = reward + self.gamma * self.values[next_state[0], next_state[1], self.out.index(next_state[2])]
                            max_value = max(max_value, next_value)
                        new_values[i, j, self.out.index(k)] = max_value
            print("Iteration:", iteration)
            error = np.max(np.abs(new_values - self.values))
            print("Error:", error)
            if error < 0.1:
                print("Converged!")
                break
            self.values = np.copy(new_values)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in self.out:
                    max_action = None
                    max_value = float('-inf')
                    for action in self.actions:
                        next_state = self.transition((i, j, k), action)
                        reward = self.main_reward((i, j, k), action, next_state)
                        next_value = reward + self.gamma * self.values[next_state[0], next_state[1], self.out.index(next_state[2])]
                        if next_value > max_value:
                            max_value = next_value
                            max_action = action
                    self.policy[i, j, self.out.index(k)] = self.actions.index(max_action)