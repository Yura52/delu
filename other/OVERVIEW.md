## Introduction
The document is a high-level overview of Zero. It covers some modules, classes and functions in a comfortable-to-read order. Pay attention to both code and comments. For fully working examples, see [examples](../examples).

Enjoy!

## `zero.all`
The module contains all content from all submodules. It is neither "better" nor "worse" to use this module instead of explicit imports from submodules, it is completely up to a user. Just keep in mind that if *all* submodules you need do not import `torch` (or any other heavy libraries) under the hood, then they will be imported faster individually then via `zero.all`.

*Example*:
```python
import zero.all as zero
zero.foo()
zero.bar()

# or
from zero.all import foo, bar
foo()
bar()
```

## `zero.flow`

#### `Flow`
The class simplifies managing for-loops:
- automatic management of the `epoch` and `iteration` variables
- allows to customize the size of epoch
- allows to change the underlying data loader on the fly
- enables useful patterns
- (not implemented: [issue](https://github.com/Yura52/zero/issues/6)) allows to dump and restore loop's state: epoch, iteration, etc.

*Before*:
```python
loader = DataLoader(...)
iteration = 0
for epoch in range(max_epoch):
    for x in loader:
        iteration += 1
        print('Epoch:', epoch, 'Iteration:', iteration)
        ...
    if need_new_loader():
        assert False, 'It is possible, but not convenient'
```

*After v1*:
```python
from zero.flow import Flow

flow = Flow(DataLoader(...))  # any kind of iterable is allowed
# loader is available as flow.loader
# "while" instead of "for epoch in range(max_epoch)":
# - is more friendly to resuming after loading a checkpoint (i.e. starting from a non-zero epoch)
# - enables flexible termination patterns (see zero.progress.ProgressTracker)
while flow.increment_epoch(max_epoch):
    for x in flow.data():  # or: `for x in flow.data(custom_epoch_size)`
        print('Epoch:', flow.epoch, 'Iteration:', flow.iteration)
        ...
    if need_new_loader():
        flow.set_loader(other_iterable)
```

*After v2*:
```python
# endless loop
for x in flow.data(math.inf):
    ...
    if flow.iteration % frequency == 0:
        ...
```

*After v3*:
```python
# endless loop
while True:
    x = flow.next()
    ...
    if flow.iteration % frequency == 0:
        ...
```

## `zero.training`

#### `ibackward`
The function combines two calls: .backward() and .item()

*Before*:
```python
loss = loss_fn(...)
loss.backward()
loss = loss.item()
```

*After*:
```python
from zero.training import ibackward

loss = ibackward(loss_fn(...))
```

#### `Eval`

*Before*:
```python
model.eval()
with torch.no_grad():
    ...
```

*After*:
```python
from zero.training import Eval

# also reverts the model's training status in __exit__ to the previous state
with Eval(model):
    ...
```

## `zero.metrics`

#### `Metric`
A simple base class for creating metrics.

*Example*:
```python
from zero.metrics import Metric

class Accuracy(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.n_objects = 0
        self.n_correct = 0
    
    def update(self, y_pred, y):
        self.n_objects += len(y)
        self.n_correct += (y_pred == y).sum().item()

    def compute(self):
        assert self.n_objects
        return self.n_correct / self.n_objects

metric_fn = Accuracy(...)
...
with Eval(model), metric_fn:  # metric_fn.reset() is called in __enter__ and __exit__
    for X, y in val_loader:
        metric_fn.update(model(X), y)
    metrics = metric_fn.compute()
```

#### `MetricsList`, `MetricsDict`
Containers for metrics (with support of [`ignite.metrics`](https://pytorch.org/ignite/metrics.html#how-to-create-a-custom-metric)).

*Example*:
```python
from ignite.metrics import Accuracy, Precision, Recall
from zero.metrics import MetricsDict

metric_fn = MetricsDict({
    'accuracy': Accuracy(...),
    'precision': Precision(...),
    'recall': Recall(...),
    'custom_metric': CustomMetric(...)  # derived from zero.metrics.Metric
})

with Eval(model), metric_fn:
    for X, y in val_loader:
        metric_fn.update((model(X), y))  # Ignite metrics expect tuples as input
    metrics = metric_fn.compute()  # {'accuracy': <float>, 'precision': <float>, ...}
```

#### `IgniteMetric`
A wrapper for metrics from [`ignite.metrics`](https://pytorch.org/ignite/metrics.html#how-to-create-a-custom-metric) that adds some functionality (e.g. support of the `with` operator).

*Example*:
```python
from ignite.metrics import Accuracy
from zero.metrics import IgniteMetric

metric = IgniteMetric(Accuracy(...))
```

## `zero.optim`
The module adds extra functionality to optimizers (*all original functionality is left unchanged*).

*Before*:
```python
from torch.optim import SGD

optimizer = SGD(...)
optimizer.zero_grad()
...
<backward>
...
optimizer.step()
```

*After*:
```python
from zero.optim import SGD

optimizer = SGD(...)  # no changes
with optimizer:
    ...
    <backward>
    ...
assert issubclass(zero.optim.SGD, torch.optim.SGD)
```

In case of optimizers not from PyTorch:
```python
from some_library import SuperOptimizer
from zero.optim import make_zero_optimizer

SuperOptimizer = make_zero_optimizer(SuperOptimizer)

optimizer = SuperOptimizer(model.parameters(), ...)
with optimizer:
    ...
```

## `zero.data`

#### `NamedTensorDataset`
The same as `torch.utils.data.TensorDataset`, but the data is named.

*Before*:
```python
dataset = TensorDataset(X, y)
assert dataset.tensors[0] is X
assert dataset.tensors[1] is y
for batch in DataLoader(dataset, ...):
    print('X:', batch[0])
    print('y:', batch[1])
```

*After*:
```python
from zero.data import NamedTensorDataset

dataset = NamedTensorDataset(X, y, names=['X', 'y'])
# or
dataset = NamedTensorDataset.from_dict({'X': X, 'y': y})
assert dataset.X is X
assert dataset.y is y
for batch in DataLoader(dataset, ...):
    # batch is a named tuple
    print('X:', batch.X)
    print('y:', batch.y)
```

#### `Enumerate`
Solves [this](https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948) problem.

*Before*:
```python
class MyDataset:
    def __getitem__(self, idx):
        return idx, data
```

*After*:
```python
from zero.data import Enumerate

class MyDataset:
    def __getitem__(self, idx):
        return data  # no need to return the index anymore

loader = DataLoader(Enumerate(my_dataset))
for idx, batch in loader:
    ...
```

#### `iloader`
Returns a DataLoader over batch indices. **The shuffling logic is fully delegated to native PyTorch DataLoader**, i.e. no custom logic is performed under the hood. It also means that all DataLoader's `*args` and `**kwargs` are available.

*Example*:
```python
idx_loader = iloader(dataset_size, batch_size)
# All arguments of DataLoader.__init__ can be passed to the function:
# iloader(dataset_size, batch_size, shuffle=True, drop_last=True)
while True:
    # indices are reshuffled once per for-loop, since it is a native PyTorch DataLoader
    for batch_idx in idx_loader:
        batch = data[batch_idx]
        ...
```

#### `iter_batches`
A more efficient alternative to PyTorch DataLoaders for tensors-based data. The function uses `iloader` under the hood, i.e. **the shuffling logic is fully delegated to native PyTorch DataLoader**.

```python
from zero.data import iter_batches

# every data below is valid input to the function
data = X
data = (X, y)
data = MyNamedTuple(X, y)
data = {'X': X, 'y': y}
data = torch.utils.data.TensorDataset(X, y)
data = zero.data.NamedTensorDataset(X, y, names=['X', 'y'])
for batch in iter_batches(data, batch_size):
    ...
```

## `zero.map_concat`

#### `concat`
If you have a function (**or a model**) that is applied to batches, `concat` will help you to combine a list (or iterable) of batch results in one result for the whole data. It can process batch results that are tensors, numpy-arrays, tuples, dictionaries, lists of arbitrary data and reasonable combinations of the mentioned types. If your workflow involves moving data between devices, use `concat` in combination with [`dmap`](#dmap).

*Before*:
```python
# (A) if model_or_fn(x) is a tensor:
result = torch.cat(tuple(map(model_or_fn, batches)))

# (B) elif model_or_fn(x) is a numpy-array:
result = np.concatenate(tuple(map(model_or_fn, batches)))

# (C) else:
def complex_model_or_fn(...):
    return {
        'a': batch_tensor,  # B x N
        'b': batch_numpy_array,  # B x N
        'c': list_of_integers  # len == B
    }
batch_results = list(map(complex_model_or_fn, batches))
result = <custom merge of batch results>
```

*After*:
```python
from zero.map_concat import concat

# The same code for all cases (A, B, C), no need to convert the input to tuple
result = concat(map(model_or_fn, batches)) 
```

*In combination with `zero.data.iter_batches`*:
```python
from zero.data import iter_batches
from zero.map_concat import concat

result = concat(map(f, iter_batches(data, batch_size)))
```

#### `dmap`
Devices-aware version of `map`.

*Before*:
```python
batch_results = []
for x in batches:
    x = to_device(x, in_device)
    x = to_in_device(x)
    batch_result = model(x)
    batch_results.append(to_device(batch_result, out_device))
result = concat(batch_results)
```

*After*:
```python
result = concat(dmap(model, batches, in_device, out_device))
```

## `zero.progress`

#### `ProgressTracker`
- helps with Early Stopping ("no progress for too many updates")
- tracks the best score
- (not implemeted: [issue](https://github.com/Yura52/zero/issues/5)) allows to dump and restore tracker's state

*Example*:
```python
from zero.flow import Flow
from zero.progress import ProgressTracker

progress = ProgressTracker(5, 0.0001)  # ProgressTracker(patience, tolerance)
flow = Flow(...)
# early stopping
while not progress.fail and flow.increment_epoch(max_epoch):
    for batch in flow.data():
        ...
    progress.update(calculate_score())
    if progress.success:
        print('New best score:', progress.best_score)
    elif progress.fail:
        print('No progress for more than 5 updates')
    else:
        print('The best score was not updated, but it is still not a fail, because the patience is big enough')
```

## `zero.time`

#### `Timer`
Time-management as simple as two methods.
- `Timer.start()` for starting/resuming
- `Timer.stop()` for pausing
- (not implemented: [issue](https://github.com/Yura52/zero/issues/7)) allows to dump and restore timer's state

```python
timer = Timer()
for x in data:
    timer.start()
    train_step(x)
    print('Total training time:', timer())

    if need_validation():
        timer.stop()
        validation()
```

#### `format_seconds`
*Before*:
```python
from time import strftime, gmtime
print(strftime('%Hh %Mm %Ss', gmtime(timer())))
```

*After*:
```python
from zero.time import format_seconds
print(format_seconds(timer()))  # The format is customizable, default: '%Hh %Mm %Ss'
```

## `zero.hardware`

#### `to_device`
Painlessly transfer tensor-based data between devices.

*Example*:
```python
from zero.hardware import to_device

# works for all data given below
data = X
data = (X, y)
data = {
    'a': [x1, (x2, x3)],
    'b': {'c': {'d': [[[x4]]]}},
    'c': MyNamedTuple(x5, {'d': x6}),
}
new_data = to_device(data, 'cuda')
```

#### `free_memory`
Runs the garbage collector, frees all unused RAM and GPU memory.

*Example*:
```python
from zero.hardware import free_memory

free_memory()
```

#### `get_gpu_info`
A handy function for getting information about GPUs.

*Example*:
```python
from zero.hardware import get_gpu_info

print(get_gpu_info())
# example
[
    {
        'total': 11019.4375,
        'used': 0.0625,
        'free': 11019.375,
        'used%': 0.0005671795860723381,
        'free%': 99.99943282041393,
        'util%': 0.0,
    },
    {
        'total': 11016.9375,
        'used': 0.0625,
        'free': 11016.875,
        'used%': 0.0005673082923453092,
        'free%': 99.99943269170765,
        'util%': 0.0,
    },
]
```

## `zero.random`

#### `set_seed_everywhere`
Simplifies reproducibility and following good practices.
- sets random seed for the following modules: `random`, `np.random`, `torch`, `torch.cuda`
- if seed is omitted, a high-quality seed is generated
- whatever seed is used, it is logged so you can reuse it to reproduce your experiment
- a new-style numpy-generator is returned (note: functions from `np.random` is not the recommended way of working with randomness in numpy anymore, see [this](https://numpy.org/doc/stable/reference/random/index.html) document for details; TL;DR is "new style provides better API and better statistical properties")

*Example*:
```python
from zero.random import set_seed_everywhere

new_style_numpy_rng = set_seed_everywhere(my_seed)
# or
new_style_numpy_rng = set_seed_everywhere()
# in both cases, prints something like this:
# Seed sequence: <the seed used> (see zero.random.set_seed_everywhere)'
```

## `zero.io`
Shortcuts for reading/writing JSON, JSONL, Pickle.

*Before*:
```python
import json

with open('data.json') as f:
    x = json.load(f)

with open('data.json', 'w') as f:
    json.dump(x, f)
```

*After*:
```python
from zero.io import load_json, dump_json

x = load_json('data.json')
dump_json(x, 'data.json')
```

The same goes for JSONL and Pickle.
