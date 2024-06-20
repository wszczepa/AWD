import pygame
import numpy as np
from queue import PriorityQueue
from collections import deque
import random

WINDOW_SIZE = 1200
GRID_SIZE = 20
cell_size = WINDOW_SIZE // GRID_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (169, 169, 169)

pygame.init()
win = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 40))
pygame.display.set_caption("Najkrótsza ścieżka w labiryncie")
font = pygame.font.SysFont("Arial", 20)

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def h_chebyshev(start, end):
    return max(abs(start[0] - end[0]), abs(start[1] - end[1]))

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    return path

def draw_grid(path=[], current=None, grid=[], start=None, end=None, highlight_color=None, visited=set()):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = WHITE if grid[j][i] == 0 else BLACK
            if (i, j) in visited and (i, j) not in path:
                color = DARK_GRAY
            if (i, j) in path:
                color = GREEN
            if start and (i, j) == start:
                color = RED
            if end and (i, j) == end:
                color = BLUE
            if current and (i, j) == current:
                color = YELLOW
            pygame.draw.rect(win, color, (i * cell_size, j * cell_size + 60, cell_size, cell_size))
            if highlight_color and (i, j) == highlight_color:
                pygame.draw.rect(win, GRAY, (i * cell_size, j * cell_size + 60, cell_size, cell_size), 2)

def draw_instructions():
    instructions = "Wybierz na klawiaturze: 'S' - start, 'K' - koniec, 'P' - ścieżki, 'X' - ściany. Kliknij myszką, aby ustawić. SPACE - zakończ rysowanie. Q - restart."
    algorithms = "1 - Algorytm BFS, 2 - Algorytm A*, 3 - Algorytm DFS, 4 - Algorytm A* (chebyshev)"
    text_surface = font.render(instructions, True, WHITE)
    text_surface2 = font.render(algorithms, True, WHITE)
    win.blit(text_surface, (10, 10))
    win.blit(text_surface2, (10, 30))


def a_star_search_chebyshev(start, end, grid):
    
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = { (i, j): float("inf") for i in range(GRID_SIZE) for j in range(GRID_SIZE) }
    g_score[start] = 0
    f_score = { (i, j): float("inf") for i in range(GRID_SIZE) for j in range(GRID_SIZE) }
    f_score[start] = h_chebyshev(start, end)
    
    open_set_hash = {start}
    visited = set()

    while not open_set.empty():
        current = open_set.get()[2]
        open_set_hash.remove(current)
        visited.add(current)

        draw_grid(reconstruct_path(came_from, current), current, grid, start, end, visited=visited)
        pygame.display.flip()
        pygame.time.delay(100)
        if current == end:
            return True, came_from, visited

        for neighbor in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            neighbor = (current[0] + neighbor[0], current[1] + neighbor[1])

            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE and grid[neighbor[1]][neighbor[0]] == 0:
                temp_g_score = g_score[current] + 1

                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + h_chebyshev(neighbor, end)
                    if neighbor not in open_set_hash:
                        count += 1
                        open_set.put((f_score[neighbor], count, neighbor))
                        open_set_hash.add(neighbor)
    return False, {}, visited


def a_star_search(start, end, grid):
    
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = { (i, j): float("inf") for i in range(GRID_SIZE) for j in range(GRID_SIZE) }
    g_score[start] = 0
    f_score = { (i, j): float("inf") for i in range(GRID_SIZE) for j in range(GRID_SIZE) }
    f_score[start] = h(start, end)
    
    open_set_hash = {start}
    visited = set()

    while not open_set.empty():
        current = open_set.get()[2]
        open_set_hash.remove(current)
        visited.add(current)

        draw_grid(reconstruct_path(came_from, current), current, grid, start, end, visited=visited)
        pygame.display.flip()
        pygame.time.delay(100)
        if current == end:
            return True, came_from, visited

        for neighbor in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            neighbor = (current[0] + neighbor[0], current[1] + neighbor[1])

            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE and grid[neighbor[1]][neighbor[0]] == 0:
                temp_g_score = g_score[current] + 1

                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + h(neighbor, end)
                    if neighbor not in open_set_hash:
                        count += 1
                        open_set.put((f_score[neighbor], count, neighbor))
                        open_set_hash.add(neighbor)
    return False, {}, visited


def dfs_search(start, end, grid):
    stack = [start]
    came_from = {start: None}
    visited = set([start])

    while stack:
        current = stack.pop()
        draw_grid(reconstruct_path(came_from, current), current, grid, start, end, visited=visited)
        pygame.display.flip()
        pygame.time.delay(100)
        if current == end:
            return True, came_from, visited

        for neighbor in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            neighbor = (current[0] + neighbor[0], current[1] + neighbor[1])

            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE and grid[neighbor[1]][neighbor[0]] == 0:
                if neighbor not in visited:
                    stack.append(neighbor)
                    came_from[neighbor] = current
                    visited.add(neighbor)

    return False, {}, visited


def bfs(start, end, grid):
    queue = deque([start])
    came_from = {start: None}
    visited = set([start])

    while queue:
        current = queue.popleft()
        draw_grid(reconstruct_path(came_from, current), current, grid, start, end, visited=visited)
        pygame.display.flip()
        pygame.time.delay(100)
        if current == end:
            return True, came_from, visited
        
        for neighbor in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            neighbor = (current[0] + neighbor[0], current[1] + neighbor[1])

            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE and grid[neighbor[1]][neighbor[0]] == 0:
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

    return False, {}, visited

def generate_maze(grid):
    stack = [(0, 0)]
    grid[0][0] = 0

    while stack:
        x, y = stack[-1]
        grid[y][x] = 0
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        random.shuffle(neighbors)
        found = False
        for nx, ny in neighbors:
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny][nx] == 1:
                if sum(grid[ny + dy][nx + dx] == 0 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]) <= 1:
                    stack.append((nx, ny))
                    found = True
                    break
        if not found:
            stack.pop()

def main():
    i = 0
    algorytm = None
    while True:
        if i == 0:
            grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)
            start = None
            end = None
            mode = None
        drawing = True
        path = []
        visited = set()
        while drawing:
            if i == 0:
                win.fill(BLACK)
                i = i + 1
            
            draw_instructions()
            draw_grid(grid=grid, start=start, end=end, highlight_color=None)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        mode = 'start'
                    elif event.key == pygame.K_k:
                        mode = 'end'
                    elif event.key == pygame.K_p:
                        mode = 'path'
                    elif event.key == pygame.K_x:
                        mode = 'wall'
                    elif event.key == pygame.K_1:
                        algorytm = 'bfs'
                        print("Wybrano algorytm bfs")
                    elif event.key == pygame.K_2:
                        algorytm = 'a*'
                        print("Wybrano algorytm a*")
                    elif event.key == pygame.K_SPACE:
                        drawing = False
                    elif event.key == pygame.K_3:
                        algorytm = 'dfs_search'
                        print("Wybrano algorytm dfs_search")
                    elif event.key == pygame.K_4:
                        algorytm = 'a_star_search_chebyshev'
                        print("Wybrano algorytm a_star_search_chebyshev")
                    elif event.key == pygame.K_c:
                        grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)
                        start = None
                        end = None
                        mode = None

            highlight_color = None
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                x, y = pos[0] // cell_size, (pos[1] - 40) // cell_size
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    if mode == 'start':
                        start = (x, y)
                        grid[y][x] = 0
                    elif mode == 'end':
                        end = (x, y)
                        grid[y][x] = 0
                    elif mode == 'path':
                        if (x, y) != start and (x, y) != end:
                            grid[y][x] = 0
                    elif mode == 'wall':
                        if (x, y) != start and (x, y) != end:
                            grid[y][x] = 1
                if mode == 'start':
                    highlight_color = (x, y)
                elif mode == 'end':
                    highlight_color = (x, y)

            draw_grid(grid=grid, start=start, end=end, highlight_color=highlight_color)
            pygame.display.flip()
        if algorytm == 'bfs':     
            path_found, came_from, visited = bfs(start, end, grid)
        elif algorytm == 'a*':
            path_found, came_from, visited = a_star_search(start, end, grid)
        elif algorytm == 'dfs_search':
            path_found, came_from, visited = dfs_search(start, end, grid)
        elif algorytm == 'a_star_search_chebyshev':
            path_found, came_from, visited = a_star_search_chebyshev(start, end, grid)
        if path_found:
            path = reconstruct_path(came_from, end)
            current = start
            for step in path:
                draw_grid(path, current, grid, start, end, visited=visited)
                pygame.time.delay(100)
                current = step
            draw_grid(path, None, grid, start, end, visited=visited)
            
      
            path_length = len(path)
            visited_non_path = len([cell for cell in visited if cell not in path])
            if algorytm == 'a*' or "a_star_search_chebyshev":
                print(f"Długość najkrótszej ścieżki: {path_length}")
                print(f"Liczba odwiedzonych pól niebędących częścią ścieżki: {visited_non_path}")
            elif algorytm == "bfs" or "dfs_search":
                print(f"Długość najkrótszej ścieżki: {path_length}")
                print(f"Liczba odwiedzonych pól niebędących częścią ścieżki: {visited_non_path}")
        else:
            print("Ścieżka nie została znaleziona.")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                        # Usuwanie szarych i zielonych pól, zastąpienie ich białymi
                        for (i, j) in path:
                            grid[j][i] = 0
                        for (i, j) in visited:
                            grid[j][i] = 0

main()
