import numpy as np
import hungarian_optimizer
ho = hungarian_optimizer.HungarianOptimizer()

# test case 1
# 0.1, 1.0
# 1.0, 0.1
# matches:
# 0->0, 1->1
costs = np.array([[0.1, 1.0], [1.0, 0.1]])
assignments = ho.minimize(costs)
assert np.all(assignments == np.array([[0, 0], [1, 1]], np.int8))

# test case 2
# 0.0, 0.0
# 0.0, 0.0
# matches:
# 0->0, 1->1
costs = np.array([[0.0, 0.0], [0.0, 0.0]])
assignments = ho.minimize(costs)
assert np.all(assignments == np.array([[0, 0], [1, 1]], np.int8))

# test case 3
# 3.0, 3.0
# 3.0, 3.0
# matches:
# 0->0, 1->1
costs = np.array([[3.0, 3.0], [3.0, 3.0]])
assignments = ho.minimize(costs)
assert np.all(assignments == np.array([[0, 0], [1, 1]], np.int8))

# test case 4
# 4.7, 3.8, 1.0, 2.0
# 4.1, 3.0, 2.0, 3.0
# 1.0, 2.0, 4.7, 4.9
# matches:
# 0->2, 1->1, 2->0
costs = np.array([[4.7, 3.8, 1.0, 2.0], [4.1, 3.0, 2.0, 3.0], [1.0, 2.0, 4.7, 4.9]])
assignments = ho.minimize(costs)
assert np.all(assignments == np.array([[0, 2], [1, 1], [2, 0]], np.int8))

# test case 5
# 4.7, 3.8, 1.0
# 4.1, 3.0, 2.0
# 1.0, 2.0, 4.7
# 3.2, 2.1, 0.5
# matches:
# 0->2, 2->0, 3->1
costs = np.array([[4.7, 3.8, 1.0], [4.1, 3.0, 2.0], [1.0, 2.0, 4.7], [3.2, 2.1, 0.5]])
assignments = ho.minimize(costs)
assert np.all(assignments == np.array([[0, 2], [2, 0], [3, 1]], np.int8))

# test case 6: empty
costs = np.empty([0, 0])
assignments = ho.minimize(costs)
assert np.all(assignments == np.array([], np.int8).reshape([-1, 2]))

# test case 7
# 0.1, 1.0
# 1.0, 0.1
# matches:
# 0->1, 1->0
costs = np.array([[0.1, 1.0], [1.0, 0.1]])
assignments = ho.maximize(costs)
assert np.all(assignments == np.array([[0, 1], [1, 0]], np.int8))

# test case 8
# 0.0, 0.0
# 0.0, 0.0
# matches:
# 0->0, 1->1
costs = np.array([[0.0, 0.0], [0.0, 0.0]])
assignments = ho.maximize(costs)
assert np.all(assignments == np.array([[0, 0], [1, 1]], np.int8))

# test case 9
# 3.0, 3.0
# 3.0, 3.0
# matches:
# 0->0, 1->1
costs = np.array([[3.0, 3.0], [3.0, 3.0]])
assignments = ho.maximize(costs)
assert np.all(assignments == np.array([[0, 0], [1, 1]], np.int8))

# test case 10
# 4.7, 3.8, 1.0, 2.0
# 4.1, 3.0, 2.0, 3.0
# 1.0, 2.0, 4.7, 4.9
# matches:
# 0->1, 1->0, 2->3
costs = np.array([[4.7, 3.8, 1.0, 2.0], [4.1, 3.0, 2.0, 3.0], [1.0, 2.0, 4.7, 4.9]])
assignments = ho.maximize(costs)
assert np.all(assignments == np.array([[0, 1], [1, 0], [2, 3]], np.int8))

# test case 11
# 4.7, 3.8, 1.0
# 4.1, 3.0, 2.0
# 1.0, 2.0, 4.7
# 3.2, 2.1, 0.5
# matches:
# 0->1, 1->0, 2->2
costs = np.array([[4.7, 3.8, 1.0], [4.1, 3.0, 2.0], [1.0, 2.0, 4.7], [3.2, 2.1, 0.5]])
assignments = ho.maximize(costs)
assert np.all(assignments == np.array([[0, 1], [1, 0], [2, 2]], np.int8))

# test case 12: empty
costs = np.empty([0, 0])
assignments = ho.maximize(costs)
assert np.all(assignments == np.array([], np.int8).reshape([-1, 2]))

