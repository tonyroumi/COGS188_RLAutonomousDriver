import numpy as np
import heapq


def astar_search(grid, start, goal):
    """
    Performs A* search on a grid to compute path.
    
    Parameters:
        grid (numpy.ndarray): 2D numpy array (0 for free, nonzero for obstacles).
        start (tuple): The starting cell (row, col).
        goal (tuple): The goal cell (row, col).
    
    Returns:
        list: The path from start to goal as a list of (row, col) tuples, or None if no path is found.
    """
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (np.linalg.norm(np.array(start) - np.array(goal)), start))
    came_from = {}
    g_score = {start: 0}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 0:
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + np.linalg.norm(np.array(neighbor) - np.array(goal))
                    heapq.heappush(open_set, (f_score, neighbor))
    
    return None 