# Hungarian Optimizer

This is a replication of [Baidu Apollo](https://www.github.com/ApolloAuto/apollo/)'s [hungarian optimizer](https://www.github.com/ApolloAuto/apollo/blob/v7.0.0/modules/perception/common/graph/hungarian_optimizer.h).

## How to use?

```python
import hungarian_optimizer
ho = hungarian_optimizer.HungarianOptimizer()
costs = ... # your costs matrix, the shape should be m * n
# to minimize
assignments = ho.minimize(costs)
# to maximize
assignemtns = ho.maximize(costs)
```
You can also refer to [test.py](./tests.py) to see the examples.
