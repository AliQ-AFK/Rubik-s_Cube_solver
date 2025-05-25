import random
import collections
import heapq # neeeded in A* priority queue
import time

# --- Configuration for 2x2x2 Cube ---
N_DIMENSION = 2
# Adjust scramble moves for testing; fewer moves are easier/faster to solve
SCRAMBLE_MOVES_COUNT = 4  # e.g., 3-6 moves for 2x2x2 is good for quick tests
DFS_DEPTH_LIMIT = 11     # Max solution length for 2x2x2 is 11 in quarter turns

# --- 1. Cube State Representation and Initial Cube Generation ---

def generate_solved_immutable_cube(n):
    """
    Generates an immutable representation of a solved N x N x N Rubik's Cube.
    Returns a tuple of faces (Up, Down, Left, Front, Right, Back), 
    where each face is a tuple of rows, and each row is a tuple of sticker strings.
    """
    if n < 1: return None # Should ideally be n>=2 for puzzles

    # Standard colors and face order
    face_color_map = {'U': 'W', 'D': 'Y', 'L': 'O', 'F': 'G', 'R': 'R', 'B': 'B'}
    standard_face_order_ids = ['U', 'D', 'L', 'F', 'R', 'B']
    
    all_faces_of_the_cube = []
    for face_id_char in standard_face_order_ids:
        current_face_color = face_color_map[face_id_char]
        a_single_face = []
        for _ in range(n): 
            a_single_row = tuple([current_face_color] * n) # Row to tuple
            a_single_face.append(a_single_row)
        all_faces_of_the_cube.append(tuple(a_single_face)) # Face to tuple
    return tuple(all_faces_of_the_cube) # Final cube to tuple

# --- Helper Functions for State Mutability (used inside move functions) ---

def _state_to_mutable_lists(state_tuple):
    """Converts immutable tuple state to a mutable list of lists of lists."""
    return [list(map(list, face)) for face in state_tuple]

def _mutable_lists_to_state_tuple(mutable_faces):
    """Converts mutable list state back to the immutable tuple state."""
    return tuple(map(lambda face_list: tuple(map(tuple, face_list)), mutable_faces))

# --- 2. 2x2x2 Specific Move Functions ---
# (U, D, L, R, F, B) and their primes (')
# Each function takes the immutable state and returns a new immutable state.
# Face indices: U=0, D=1, L=2, F=3, R=4, B=5

def rotate_U_clockwise_2x2(state):
    faces = _state_to_mutable_lists(state)
    # Rotate U face
    u_face = faces[0]; temp = u_face[0][0]
    u_face[0][0]=u_face[1][0]; u_face[1][0]=u_face[1][1]; u_face[1][1]=u_face[0][1]; u_face[0][1]=temp
    # Rotate side top rows: F->R->B->L->F (indices 3,4,5,2)
    temp_row = faces[3][0][:] # Important: copy the row
    faces[3][0]=faces[2][0][:]; faces[2][0]=faces[5][0][:]; faces[5][0]=faces[4][0][:]; faces[4][0]=temp_row
    return _mutable_lists_to_state_tuple(faces)

def rotate_U_prime_2x2(state):
    faces = _state_to_mutable_lists(state)
    u_face = faces[0]; temp = u_face[0][0]
    u_face[0][0]=u_face[0][1]; u_face[0][1]=u_face[1][1]; u_face[1][1]=u_face[1][0]; u_face[1][0]=temp
    temp_row = faces[3][0][:]
    faces[3][0]=faces[4][0][:]; faces[4][0]=faces[5][0][:]; faces[5][0]=faces[2][0][:]; faces[2][0]=temp_row
    return _mutable_lists_to_state_tuple(faces)

def rotate_D_clockwise_2x2(state):
    faces = _state_to_mutable_lists(state)
    d_face = faces[1]; temp = d_face[0][0]
    d_face[0][0]=d_face[1][0]; d_face[1][0]=d_face[1][1]; d_face[1][1]=d_face[0][1]; d_face[0][1]=temp
    # Rotate side bottom rows: F->L->B->R->F (indices 3,2,5,4)
    temp_row = faces[3][1][:]
    faces[3][1]=faces[4][1][:]; faces[4][1]=faces[5][1][:]; faces[5][1]=faces[2][1][:]; faces[2][1]=temp_row
    return _mutable_lists_to_state_tuple(faces)

def rotate_D_prime_2x2(state):
    faces = _state_to_mutable_lists(state)
    d_face = faces[1]; temp = d_face[0][0]
    d_face[0][0]=d_face[0][1]; d_face[0][1]=d_face[1][1]; d_face[1][1]=d_face[1][0]; d_face[1][0]=temp
    temp_row = faces[3][1][:]
    faces[3][1]=faces[2][1][:]; faces[2][1]=faces[5][1][:]; faces[5][1]=faces[4][1][:]; faces[4][1]=temp_row
    return _mutable_lists_to_state_tuple(faces)

def rotate_F_clockwise_2x2(state):
    faces = _state_to_mutable_lists(state)
    f_face = faces[3]; temp = f_face[0][0]
    f_face[0][0]=f_face[1][0]; f_face[1][0]=f_face[1][1]; f_face[1][1]=f_face[0][1]; f_face[0][1]=temp
    # Sides: U_bot -> R_left_col -> D_top_row -> L_right_col -> U_bot
    # U(0), D(1), L(2), R(4)
    # Save U's bottom row stickers
    u_s0, u_s1 = faces[0][1][0], faces[0][1][1]
    faces[0][1][0]=faces[2][1][1]; faces[0][1][1]=faces[2][0][1] # U_bot <- L_right_col (L_br, L_tr)
    faces[2][1][1]=faces[1][0][1]; faces[2][0][1]=faces[1][0][0] # L_right_col <- D_top_row (D_tr, D_tl)
    faces[1][0][1]=faces[4][0][0]; faces[1][0][0]=faces[4][1][0] # D_top_row <- R_left_col (R_tl, R_bl)
    faces[4][0][0]=u_s0; faces[4][1][0]=u_s1                # R_left_col <- Original U_bot (U_bl, U_br)
    return _mutable_lists_to_state_tuple(faces)

def rotate_F_prime_2x2(state):
    faces = _state_to_mutable_lists(state)
    f_face = faces[3]; temp = f_face[0][0]
    f_face[0][0]=f_face[0][1]; f_face[0][1]=f_face[1][1]; f_face[1][1]=f_face[1][0]; f_face[1][0]=temp
    # Sides: U_bot -> L_right_col -> D_top_row -> R_left_col -> U_bot (reversed)
    u_s0, u_s1 = faces[0][1][0], faces[0][1][1]
    faces[0][1][0]=faces[4][0][0]; faces[0][1][1]=faces[4][1][0] # U_bot <- R_left_col
    faces[4][1][0]=faces[1][0][0]; faces[4][0][0]=faces[1][0][1] # R_left_col <- D_top_row (reversed elements)
    faces[1][0][0]=faces[2][0][1]; faces[1][0][1]=faces[2][1][1] # D_top_row <- L_right_col
    faces[2][1][1]=u_s0; faces[2][0][1]=u_s1                # L_right_col <- Original U_bot (reversed elements)
    return _mutable_lists_to_state_tuple(faces)

def rotate_R_clockwise_2x2(state):
    faces = _state_to_mutable_lists(state)
    r_face = faces[4]; temp = r_face[0][0]; r_face[0][0]=r_face[1][0]; r_face[1][0]=r_face[1][1]; r_face[1][1]=r_face[0][1]; r_face[0][1]=temp
    # Sides: U_right_col -> F_right_col -> D_right_col -> B_left_col_rev -> U_right_col
    # U(0), D(1), F(3), B(5)
    u_s0, u_s1 = faces[0][0][1], faces[0][1][1] # U_right_col (U_tr, U_br)
    faces[0][0][1]=faces[3][0][1]; faces[0][1][1]=faces[3][1][1] # U_rc <- F_rc
    faces[3][0][1]=faces[1][0][1]; faces[3][1][1]=faces[1][1][1] # F_rc <- D_rc
    faces[1][0][1]=faces[5][1][0]; faces[1][1][1]=faces[5][0][0] # D_rc <- B_left_col (B_bl, B_tl) (inverted)
    faces[5][0][0]=u_s1; faces[5][1][0]=u_s0                # B_left_col <- Original U_rc (U_br, U_tr) (inverted)
    return _mutable_lists_to_state_tuple(faces)

def rotate_R_prime_2x2(state):
    faces = _state_to_mutable_lists(state)
    r_face = faces[4]; temp = r_face[0][0]; r_face[0][0]=r_face[0][1]; r_face[0][1]=r_face[1][1]; r_face[1][1]=r_face[1][0]; r_face[1][0]=temp
    # Sides: U_rc -> B_lc_rev -> D_rc -> F_rc -> U_rc (reversed)
    u_s0, u_s1 = faces[0][0][1], faces[0][1][1]
    faces[0][0][1]=faces[5][1][0]; faces[0][1][1]=faces[5][0][0] # U_rc <- B_lc_rev (B_bl, B_tl)
    faces[5][0][0]=faces[1][1][1]; faces[5][1][0]=faces[1][0][1] # B_lc_rev <- D_rc (D_br, D_tr)
    faces[1][1][1]=faces[3][1][1]; faces[1][0][1]=faces[3][0][1] # D_rc <- F_rc
    faces[3][0][1]=u_s0; faces[3][1][1]=u_s1                # F_rc <- Original U_rc
    return _mutable_lists_to_state_tuple(faces)

def rotate_L_clockwise_2x2(state):
    faces = _state_to_mutable_lists(state)
    l_face = faces[2]; temp = l_face[0][0]; l_face[0][0]=l_face[1][0]; l_face[1][0]=l_face[1][1]; l_face[1][1]=l_face[0][1]; l_face[0][1]=temp
    # Sides: U_left_col -> B_right_col_rev -> D_left_col -> F_left_col -> U_left_col
    # U(0), D(1), F(3), B(5)
    u_s0, u_s1 = faces[0][0][0], faces[0][1][0] # U_lc (U_tl, U_bl)
    faces[0][0][0]=faces[5][0][1]; faces[0][1][0]=faces[5][1][1] # U_lc <- B_rc_rev (B_tr, B_br)
    faces[5][1][1]=faces[1][0][0]; faces[5][0][1]=faces[1][1][0] # B_rc_rev <- D_lc (D_tl, D_bl)
    faces[1][0][0]=faces[3][0][0]; faces[1][1][0]=faces[3][1][0] # D_lc <- F_lc
    faces[3][0][0]=u_s0; faces[3][1][0]=u_s1                # F_lc <- Original U_lc
    return _mutable_lists_to_state_tuple(faces)

def rotate_L_prime_2x2(state):
    faces = _state_to_mutable_lists(state)
    l_face = faces[2]; temp = l_face[0][0]; l_face[0][0]=l_face[0][1]; l_face[0][1]=l_face[1][1]; l_face[1][1]=l_face[1][0]; l_face[1][0]=temp
    # Sides: U_lc -> F_lc -> D_lc -> B_rc_rev -> U_lc (reversed)
    u_s0, u_s1 = faces[0][0][0], faces[0][1][0]
    faces[0][0][0]=faces[3][0][0]; faces[0][1][0]=faces[3][1][0] # U_lc <- F_lc
    faces[3][0][0]=faces[1][0][0]; faces[3][1][0]=faces[1][1][0] # F_lc <- D_lc
    faces[1][1][0]=faces[5][0][1]; faces[1][0][0]=faces[5][1][1] # D_lc <- B_rc_rev (B_tr, B_br)
    faces[5][0][1]=u_s1; faces[5][1][1]=u_s0                # B_rc_rev <- Original U_lc (U_bl, U_tl)
    return _mutable_lists_to_state_tuple(faces)

def rotate_B_clockwise_2x2(state):
    faces = _state_to_mutable_lists(state)
    b_face = faces[5]; temp = b_face[0][0]; b_face[0][0]=b_face[1][0]; b_face[1][0]=b_face[1][1]; b_face[1][1]=b_face[0][1]; b_face[0][1]=temp
    # Sides: U_top_row -> L_left_col_rev -> D_bot_row -> R_right_col_rev -> U_top_row
    # U(0), D(1), L(2), R(4)
    u_s0, u_s1 = faces[0][0][0], faces[0][0][1] # U_top_row (U_tl, U_tr)
    faces[0][0][0]=faces[2][0][0]; faces[0][0][1]=faces[2][1][0] # U_tr <- L_lc (L_tl, L_bl)
    faces[2][1][0]=faces[1][1][1]; faces[2][0][0]=faces[1][1][0] # L_lc <- D_br (D_br, D_bl)
    faces[1][1][1]=faces[4][1][1]; faces[1][1][0]=faces[4][0][1] # D_br <- R_rc (R_br, R_tr)
    faces[4][0][1]=u_s0; faces[4][1][1]=u_s1                # R_rc <- Original U_tr (U_tl, U_tr)
    return _mutable_lists_to_state_tuple(faces)

def rotate_B_prime_2x2(state):
    faces = _state_to_mutable_lists(state)
    b_face = faces[5]; temp = b_face[0][0]; b_face[0][0]=b_face[0][1]; b_face[0][1]=b_face[1][1]; b_face[1][1]=b_face[1][0]; b_face[1][0]=temp
    # Sides: U_tr -> R_rc_rev -> D_br -> L_lc_rev -> U_tr (reversed)
    u_s0, u_s1 = faces[0][0][0], faces[0][0][1]
    faces[0][0][0]=faces[4][0][1]; faces[0][0][1]=faces[4][1][1] # U_tr <- R_rc
    faces[4][1][1]=faces[1][1][0]; faces[4][0][1]=faces[1][1][1] # R_rc <- D_br (reversed)
    faces[1][1][0]=faces[2][1][0]; faces[1][1][1]=faces[2][0][0] # D_br <- L_lc
    faces[2][0][0]=u_s1; faces[2][1][0]=u_s0                # L_lc <- Original U_tr (reversed)
    return _mutable_lists_to_state_tuple(faces)


# --- 3. Move Dispatcher ---
ALL_MOVES_N2 = {
    "U": rotate_U_clockwise_2x2, "U'": rotate_U_prime_2x2,
    "D": rotate_D_clockwise_2x2, "D'": rotate_D_prime_2x2,
    "L": rotate_L_clockwise_2x2, "L'": rotate_L_prime_2x2,
    "R": rotate_R_clockwise_2x2, "R'": rotate_R_prime_2x2,
    "F": rotate_F_clockwise_2x2, "F'": rotate_F_prime_2x2,
    "B": rotate_B_clockwise_2x2, "B'": rotate_B_prime_2x2,
    # Optional: "2" moves can be added here as lambda s: move(move(s))
    "U2": lambda s: rotate_U_clockwise_2x2(rotate_U_clockwise_2x2(s)),
    "D2": lambda s: rotate_D_clockwise_2x2(rotate_D_clockwise_2x2(s)),
    "L2": lambda s: rotate_L_clockwise_2x2(rotate_L_clockwise_2x2(s)),
    "R2": lambda s: rotate_R_clockwise_2x2(rotate_R_clockwise_2x2(s)),
    "F2": lambda s: rotate_F_clockwise_2x2(rotate_F_clockwise_2x2(s)),
    "B2": lambda s: rotate_B_clockwise_2x2(rotate_B_clockwise_2x2(s)),
}
# For neighbor generation in search, usually use the 12 basic quarter turns
BASIC_MOVE_NAMES = ["U", "U'", "D", "D'", "L", "L'", "F", "F'", "R", "R'", "B", "B'"]

def apply_move(cube_state, move_name, n): # n is passed for consistency, but moves are N=2 specific
    if n != 2:
        print(f"Error: These move functions are for N=2 only. Cannot apply to N={n}.")
        # In a more robust system, you'd have N-dynamic moves or N-specific dispatch
        return cube_state 
    
    move_function = ALL_MOVES_N2.get(move_name)
    if move_function:
        return move_function(cube_state)
    else:
        # This case should ideally not be hit if move_name is from BASIC_MOVE_NAMES
        print(f"Warning: Move '{move_name}' is not recognized for N=2. State unchanged.")
        return cube_state

# --- 4. Scrambling Function ---
def generate_scrambled_cube(n, num_scramble_moves, solved_state_func, apply_move_func):
    if n != 2: # This scrambler uses N=2 specific apply_move
        print("Error: This scrambler is set up for N=2 only due to move functions.")
        return None

    solved_state = solved_state_func(n)
    if solved_state is None: return None

    current_cube_state = solved_state
    if num_scramble_moves <= 0: return current_cube_state
        
    print(f"\nScrambling the {n}x{n}x{n} cube with {num_scramble_moves} random moves...")
    
    scramble_sequence = []
    for _ in range(num_scramble_moves):
        random_move_name = random.choice(BASIC_MOVE_NAMES) # Use basic 12 moves for scramble
        current_cube_state = apply_move_func(current_cube_state, random_move_name, n)
        scramble_sequence.append(random_move_name)
            
    print(f"Scrambled with sequence: {' '.join(scramble_sequence)}")
    return current_cube_state, scramble_sequence

# --- 5. Heuristic Function (Sticker Manhattan Distance to Target Face for 2x2x2) ---
CANONICAL_FACE_ORDER_IDS = ['U', 'D', 'L', 'F', 'R', 'B'] 
SOLVED_COLOR_TARGET_FACE_ID_MAP = {'W':'U', 'Y':'D', 'O':'L', 'G':'F', 'R':'R', 'B':'B'}
OPPOSITE_FACE_MAP = {'U':'D', 'D':'U', 'L':'R', 'R':'L', 'F':'B', 'B':'F'}

def sticker_manhattan_heuristic_2x2(current_state, goal_state): # n=2 is fixed here
    total_distance = 0
    n_val = 2 
    
    for face_idx, current_face_data in enumerate(current_state):
        current_face_id = CANONICAL_FACE_ORDER_IDS[face_idx]
        
        for r in range(n_val):
            for c in range(n_val):
                sticker_color = current_face_data[r][c]
                target_face_id_for_sticker = SOLVED_COLOR_TARGET_FACE_ID_MAP.get(sticker_color)

                if target_face_id_for_sticker is None: continue 

                if current_face_id == target_face_id_for_sticker:
                    distance = 0
                elif OPPOSITE_FACE_MAP[current_face_id] == target_face_id_for_sticker:
                    distance = 2
                else:
                    distance = 1
                total_distance += distance
    # For 2x2x2, a turn affects 4 face stickers + 2*4=8 side stickers = 12 sticker faces.
    # Max distance can be 24*1 (if all adjacent) = 24, or if opposite 24*2=48.
    # True optimal solutions for 2x2x2 are up to 11 (quarter turns) or 14 (face turns).
    # Dividing by 4 is a common heuristic scaling for Rubik's type problems to help ensure admissibility
    # and provide a reasonable estimate in terms of moves.
    return total_distance / 4.0 

# --- 6. Search Algorithms ---

def get_neighbors_2x2(state): # Specific to N=2
    neighbors = []
    for move_name in BASIC_MOVE_NAMES: # Use the 12 basic moves
        next_state = apply_move(state, move_name, 2) # n=2
        neighbors.append((next_state, move_name))
    return neighbors

def bfs_solver_2x2(initial_state, goal_state):
    print("\n--- Starting BFS Solver (for 2x2x2) ---")
    queue = collections.deque([(initial_state, [])]) # (state, path_taken)
    visited = {initial_state}
    nodes_explored = 0
    start_time = time.time()

    while queue:
        current_state, path = queue.popleft()
        nodes_explored += 1

        if current_state == goal_state:
            end_time = time.time()
            duration = end_time - start_time
            print(f"BFS: Solution Found!")
            print(f"  Path: {' '.join(path) if path else 'Already solved!'}")
            print(f"  Path length: {len(path)}")
            print(f"  Nodes explored: {nodes_explored}")
            print(f"  Time taken: {duration:.4f} seconds")
            return path, nodes_explored, duration

        if len(path) > DFS_DEPTH_LIMIT + 2: # Optimization: Stop if path gets too long for 2x2x2
            continue

        for next_state, move_name in get_neighbors_2x2(current_state):
            if next_state not in visited:
                visited.add(next_state)
                new_path = path + [move_name]
                queue.append((next_state, new_path))
    
    end_time = time.time()
    duration = end_time - start_time
    print("BFS: No solution found (or queue exhausted).")
    print(f"  Nodes explored: {nodes_explored}")
    print(f"  Time taken: {duration:.4f} seconds")
    return None, nodes_explored, duration

def dfs_solver_2x2(initial_state, goal_state, depth_limit):
    print(f"\n--- Starting DFS Solver (for 2x2x2, Depth Limit: {depth_limit}) ---")
    # For DFS, stack stores (state, path).
    # Visited set for path cycle detection and overall explored states (can be simple set for DFS)
    stack = [(initial_state, [])] 
    visited_in_path = {} # To avoid cycles in current path: state -> path_len
    nodes_explored = 0
    start_time = time.time()

    while stack:
        current_state, path = stack.pop()
        nodes_explored += 1

        if current_state == goal_state:
            end_time = time.time()
            duration = end_time - start_time
            print(f"DFS: Solution Found!")
            print(f"  Path: {' '.join(path) if path else 'Already solved!'}")
            print(f"  Path length: {len(path)}")
            print(f"  Nodes explored: {nodes_explored}")
            print(f"  Time taken: {duration:.4f} seconds")
            return path, nodes_explored, duration

        if len(path) >= depth_limit:
            continue

        # Add neighbors (reversed for typical LIFO stack exploration)
        # Could shuffle for more randomness if desired
        neighbors = get_neighbors_2x2(current_state)
        # random.shuffle(neighbors) # Optional: explore branches in random order

        for next_state, move_name in reversed(neighbors): # explore one branch deeply
            # Simple cycle check for current path
            # A more robust visited set might be needed for general graphs,
            # but for tree-like search with depth limit, this is often sufficient.
            # To avoid re-exploring states already found via a shorter or equal path (if not optimal):
            current_path_len = len(path)
            if next_state in visited_in_path and visited_in_path[next_state] <= current_path_len +1:
                continue
            visited_in_path[next_state] = current_path_len + 1
            
            new_path = path + [move_name]
            stack.append((next_state, new_path))
            
    end_time = time.time()
    duration = end_time - start_time
    print("DFS: No solution found (stack exhausted or depth limit reached).")
    print(f"  Nodes explored: {nodes_explored}")
    print(f"  Time taken: {duration:.4f} seconds")
    return None, nodes_explored, duration

def astar_solver_2x2(initial_state, goal_state, heuristic_func):
    print("\n--- Starting A* Solver (for 2x2x2) ---")
    pq_counter = 0 # Tie-breaker for priority queue
    
    # (f_cost, g_cost, h_cost, counter, state, path)
    h_initial = heuristic_func(initial_state, goal_state)
    priority_queue = [(0 + h_initial, 0, h_initial, pq_counter, initial_state, [])] 
    
    # visited_costs stores the *lowest g_cost* found so far to reach a state
    visited_g_costs = {initial_state: 0} 
    nodes_explored = 0
    start_time = time.time()

    while priority_queue:
        f_cost, g_cost, h_cost_debug, count, current_state, path = heapq.heappop(priority_queue)
        nodes_explored += 1

        # If we found a shorter path to this state already, skip
        if g_cost > visited_g_costs.get(current_state, float('inf')):
            continue

        if current_state == goal_state:
            end_time = time.time()
            duration = end_time - start_time
            print(f"A*: Solution Found!")
            print(f"  Path: {' '.join(path) if path else 'Already solved!'}")
            print(f"  Path length (g_cost): {g_cost}")
            print(f"  Nodes explored: {nodes_explored}")
            print(f"  Time taken: {duration:.4f} seconds")
            print(f"  Final f_cost: {f_cost}, g_cost: {g_cost}, h_cost: {h_cost_debug}")
            return path, nodes_explored, duration

        if len(path) > DFS_DEPTH_LIMIT + 5: # A* might find optimal, DFS_DEPTH_LIMIT is just a safety for 2x2x2
            continue


        for next_state, move_name in get_neighbors_2x2(current_state):
            new_g_cost = g_cost + 1
            
            if new_g_cost < visited_g_costs.get(next_state, float('inf')):
                visited_g_costs[next_state] = new_g_cost
                h_val = heuristic_func(next_state, goal_state)
                new_f_cost = new_g_cost + h_val
                pq_counter += 1
                heapq.heappush(priority_queue, (new_f_cost, new_g_cost, h_val, pq_counter, next_state, path + [move_name]))
                
    end_time = time.time()
    duration = end_time - start_time
    print("A*: No solution found (priority queue exhausted).")
    print(f"  Nodes explored: {nodes_explored}")
    print(f"  Time taken: {duration:.4f} seconds")
    return None, nodes_explored, duration

# --- 7. Main Execution Block ---
if __name__ == '__main__':
    print(f"--- Rubik's Cube Solver (Focusing on N={N_DIMENSION}x{N_DIMENSION}x{N_DIMENSION}) ---")

    # Generate solved state (this is our goal)
    goal_state_2x2 = generate_solved_immutable_cube(N_DIMENSION)

    if goal_state_2x2 is None:
        print(f"Could not generate solved cube for N={N_DIMENSION}. Exiting.")
    else:
        print("\nSolved (Goal) State:")
        face_ids = ['U', 'D', 'L', 'F', 'R', 'B']
        for i, f_data in enumerate(goal_state_2x2):
            print(f"{face_ids[i]}: {f_data}")

        # Generate a scrambled cube for N=2
        initial_scrambled_state, scramble_seq = generate_scrambled_cube(
            N_DIMENSION, 
            SCRAMBLE_MOVES_COUNT,
            generate_solved_immutable_cube, # Function to generate solved state
            apply_move                  # N=2 specific move applier
        )

        if initial_scrambled_state:
            print("\nInitial Scrambled State:")
            for i, f_data in enumerate(initial_scrambled_state):
                print(f"{face_ids[i]}: {f_data}")
            
            if initial_scrambled_state == goal_state_2x2 and SCRAMBLE_MOVES_COUNT > 0:
                print("\nWarning: Scrambled state is the same as solved state. Check move/scramble logic or increase SCRAMBLE_MOVES_COUNT.")
            elif SCRAMBLE_MOVES_COUNT == 0:
                 print("\nCube is starting in solved state as SCRAMBLE_MOVES_COUNT is 0.")


            # --- Solve with BFS ---
            # bfs_solution_path, bfs_nodes, bfs_time = bfs_solver_2x2(initial_scrambled_state, goal_state_2x2)
            # print("---")
            
            # --- Solve with DFS ---
            # dfs_solution_path, dfs_nodes, dfs_time = dfs_solver_2x2(initial_scrambled_state, goal_state_2x2, DFS_DEPTH_LIMIT)
            # print("---")

            # --- Solve with A* ---
            astar_solution_path, astar_nodes, astar_time = astar_solver_2x2(
                initial_scrambled_state, 
                goal_state_2x2, 
                sticker_manhattan_heuristic_2x2
            )
            print("---")
            
            print("\n--- Summary of Results (for 2x2x2) ---")
            # print(f"Scramble: {' '.join(scramble_seq)} ({len(scramble_seq)} moves)")
            # if bfs_solution_path is not None:
            #     print(f"BFS    | Solved: Yes | Path Length: {len(bfs_solution_path):<3} | Nodes: {bfs_nodes:<7} | Time: {bfs_time:.4f}s | Path: {' '.join(bfs_solution_path)}")
            # else:
            #     print(f"BFS    | Solved: No  | Path Length: --- | Nodes: {bfs_nodes:<7} | Time: {bfs_time:.4f}s")

            # if dfs_solution_path is not None:
            #     print(f"DFS    | Solved: Yes | Path Length: {len(dfs_solution_path):<3} | Nodes: {dfs_nodes:<7} | Time: {dfs_time:.4f}s | Path: {' '.join(dfs_solution_path)}")
            # else:
            #     print(f"DFS    | Solved: No  | Path Length: --- | Nodes: {dfs_nodes:<7} | Time: {dfs_time:.4f}s")
            
            if astar_solution_path is not None:
                 print(f"A* | Solved: Yes | Path Length: {len(astar_solution_path):<3} | Nodes: {astar_nodes:<7} | Time: {astar_time:.4f}s | Path: {' '.join(astar_solution_path)}")
            else:
                print(f"A* | Solved: No  | Path Length: --- | Nodes: {astar_nodes:<7} | Time: {astar_time:.4f}s")

        else:
            print("Could not generate scrambled cube.")