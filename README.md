# Rubik's Cube Solver: A Comparative Study of Search Algorithms

This project implements and analyzes the performance of Breadth-First Search (BFS), Depth-First Search (DFS), and A* search algorithms for solving the 2x2x2 Rubik's Cube (Pocket Cube). The primary goal is to provide a comparative study of these fundamental search techniques in a computationally tractable state space, highlighting their strengths, weaknesses, and the critical role of heuristics in informed search.

## Overview

The Rubik's Cube, a classic puzzle, serves as an excellent testbed for various problem-solving strategies, particularly search algorithms[cite: 17]. This project focuses on the 2x2x2 variant to enable a detailed exploration of algorithmic behaviors and ensure verifiable correctness[cite: 2]. It addresses the challenge of state space explosion, which makes these algorithms impractical for larger cubes without significant modifications[cite: 3].

A custom heuristic, `sticker_manhattan_heuristic_2x2`, was designed for the A* algorithm, estimating the distance to the goal state by summing "face distances" of misplaced stickers and applying a scaling factor[cite: 4, 5].

## Key Features

* **State Representation**: The 2x2x2 cube's state is represented as an immutable tuple of faces, where each face is a tuple of rows, and each row is a tuple of sticker strings[cite: 90, 92, 93]. This allows states to be used in sets for visited tracking[cite: 91].
* **Move Functions**: Implements 12 basic move functions for the 2x2x2 cube (90-degree clockwise and counter-clockwise turns for each of the 6 faces)[cite: 97, 99]. Each function takes the current immutable state and returns a new one[cite: 100, 105].
* **Cube Scrambling**: A function generates scrambled cube instances by applying a sequence of random moves to the solved state[cite: 107, 108, 109].
* **Search Algorithms**:
    * **Breadth-First Search (BFS)**: Explores layer by layer, guaranteeing the shortest solution path in terms of moves[cite: 20, 117].
    * **Depth-First Search (DFS)**: Explores as far as possible along each branch before backtracking, implemented with a depth limit[cite: 19, 122].
    * **A\* Search**: An informed search algorithm using the `sticker_manhattan_heuristic_2x2` to guide its exploration towards the goal state efficiently[cite: 21, 125, 137].
* **Heuristic Function**: The `sticker_manhattan_heuristic_2x2` calculates the sum of "face distances" for each sticker to its target face (0 if correct, 1 if adjacent, 2 if opposite) and divides by a scaling factor (4) to maintain admissibility[cite: 4, 141, 146, 148, 149, 152, 153].
* **Performance Analysis**: The project includes scripts to generate performance comparison tables (`table.py`) and graphs (`graph.py`) for metrics like solution path length, nodes explored, and execution time[cite: 7, 175, 182].

## Focus on 2x2x2 Cube

The project deliberately focuses on the 2x2x2 cube due to:
* **State Space Manageability**: The 2x2x2 cube has 3,674,160 reachable configurations, making it feasible to thoroughly test algorithms and observe complete search processes[cite: 2, 24].
* **Algorithmic Fundamentals**: It retains core challenges for understanding DFS, BFS, and A\*[cite: 28, 29].
* **Heuristic Development Context**: Allows controlled testing of the custom A\* heuristic[cite: 30].

## Performance Insights

Experimental results consistently show:
* **A\***, guided by the `sticker_manhattan_heuristic_2x2`, finds optimal solutions with markedly fewer nodes explored and significantly reduced computation time compared to BFS and DFS[cite: 8]. It offers the most effective balance between solution optimality and computational efficiency for the 2x2x2 cube[cite: 12].
* **BFS** guarantees optimality but incurs high computational costs due to its exhaustive exploration[cite: 9].
* **DFS**, with a depth limit (e.g., 11, corresponding to the 2x2x2 "God's Number" in quarter turns), can be fast but doesn't inherently guarantee optimality if the limit is too restrictive or if optimal paths are missed[cite: 10, 11].

## Scalability

While effective for the 2x2x2 cube, the implemented algorithms and the simple heuristic face significant scalability limitations for larger N x N x N cubes due to the exponential growth in state space and the diminishing informativeness of such heuristics[cite: 3, 13, 14]. Solving larger cubes requires more advanced techniques like Pattern Databases or specialized algorithms (e.g., Kociemba's algorithm)[cite: 14].

## Code Structure

* `cube_solver.py`: Contains the core logic for cube representation, move functions, scrambling, search algorithms (BFS, DFS, A*), and the A* heuristic.
* `graph.py`: Utility script to generate performance comparison graphs (nodes explored, time taken vs. scramble depth) and a state space explosion graph using Matplotlib.
* `table.py`: Utility script to generate performance comparison tables and state space complexity tables as PNG images using Matplotlib and Pandas.
* `_Solving the Rubik's Cube using Search Algorithms.pdf`: The detailed research report accompanying this project.

## How to Run

1.  Ensure you have Python installed.
2.  Install necessary libraries:
    ```bash
    pip install matplotlib pandas numpy
    ```
3.  Run the main solver script:
    ```bash
    python cube_solver.py
    ```
    This will:
    * Generate a solved 2x2x2 cube state.
    * Scramble the cube using a predefined number of random moves (configurable via `SCRAMBLE_MOVES_COUNT`).
    * Attempt to solve the scrambled cube using the A\* algorithm (BFS and DFS are commented out by default in `main` but can be enabled).
    * Print the solution path, length, nodes explored, and time taken.

4.  To generate the performance graphs and tables (using placeholder data in `graph.py` and `table.py` or by integrating actual run data):
    ```bash
    python graph.py
    python table.py
    ```
    This will save PNG images of the charts in the current directory or a `tables/` subdirectory.

## Dependencies

* Python 3.x
* `collections` (standard library)
* `heapq` (standard library)
* `time` (standard library)
* `random` (standard library)
* `matplotlib` (for `graph.py` and `table.py`)
* `pandas` (for `table.py`)
* `numpy` (for `graph.py` and `table.py`)
