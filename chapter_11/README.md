#  Overview

This is a collection of Python implementations of robotics path planning algorithms.

Features:

1. Easy to read for understanding each algorithm's basic idea.

2. Widely used and practical algorithms are selected.

3. Minimum dependency.


# Requirements to run the code

For running each sample code:

- [Python 3.13.x](https://www.python.org/)
 
- [NumPy](https://numpy.org/)
 
- [SciPy](https://scipy.org/)
 
- [Matplotlib](https://matplotlib.org/)
 
- [cvxpy](https://www.cvxpy.org/) 

For development:
  
- [pytest](https://pytest.org/) (for unit tests)
  
- [pytest-xdist](https://pypi.org/project/pytest-xdist/) (for parallel unit tests)
  
- [mypy](https://mypy-lang.org/) (for type check)
  
- [sphinx](https://www.sphinx-doc.org/) (for document generation)
  
- [pycodestyle](https://pypi.org/project/pycodestyle/) (for code style check)


# How to use

1. Clone this repo.

   ```terminal
   git clone https://github.com/OpenSpaceAI/digital-intelligent-design-book.git
   ```


2. Install the required libraries.

- using conda :

  ```terminal
  conda env create -f requirements/environment.yml
  ```
 
- using pip :

  ```terminal
  pip install -r requirements/requirements.txt
  ```

3. Execute python script in each directory.

# Path Planning

## A\* algorithm

This is a 2D grid based the shortest path planning with A star algorithm.

In the animation, cyan points are searched nodes.

Its heuristic is 2D Euclid distance.

To use:
```terminal
  python your path/AStar/a_star.py
  ```

## D\* algorithm

This is a 2D grid based the shortest path planning with D star algorithm.

To use:
```terminal
  python your path/DStar/dstar.py
  ```

The animation shows a robot finding its path avoiding an obstacle using the D* search algorithm.

Reference