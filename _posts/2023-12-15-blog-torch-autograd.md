---
title: 'pytorch autograd'
date: 23-12-15
permalink: /posts/2023/12/blog-code-pytorch-autograd/
tags:
  - 深度学习基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要介绍了 PyTorch 的 autograd 机制及其具体实现方式。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">训练一个模型的范式是构造模型 $\mathcal{M}$ 和数据 $\mathcal{D} = \{x_i,y_i\}_{1:N}$，
使用模型的前向过程计算 loss：<code style="color: #B58900">l = M.forward(xi,yi)</code>，然后使用<code style="color: #B58900">l.backward()</code>计算 gradient，
最后使用<code style="color: #B58900">optim.step()</code>更新模型参数。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">那么 PyTorch 是如何计算每个参数的梯度的，即 PyTorch 的自动求导机制(<code style="color: #B58900">torch.autograd</code>)？
通俗而言，<code style="color: #B58900">torch.autograd</code>在模型 forward 的同时构造了一个 computational graph: DAG (由 Function 组成)；
其中的叶子节点表示输入数据和模型参数，而非叶子节点表示模型对这些输入数据和参数进行的数学操作(加法，乘法等)。
然后在 backward 的时候触发每个节点的 gradient 计算，并将计算完成的 gradient 存储在各自对应的<code style="color: #B58900">.grad</code>属性内。
最后 optim 的 step 执行时将每个参数的值使用对应<code style="color: #B58900">.grad</code>属性内的 gradient 进行更新计算：<code style="color: #B58900">p = p - lr * p.grad</code>。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">第一步便是如何构建计算图？首先，torch 构造的计算图是一个有向无环图(DAG)。
其中，叶子节点为输入的 tensor，包括输入的数据和参与计算的模型参数，根节点为输出的 tensor，而中间节点是模型执行的每个初等函数运算。
假设<code style="color: #B58900">a</code>和<code style="color: #B58900">b</code>是<code style="color: #B58900">torch.Tensor</code>变量，
且<code style="color: #B58900">M = lambda x, y: 3*x**3 - y**2</code>，则在<code style="color: #B58900">out=M(a,b)</code>时，<code style="color: #B58900">torch.autograd</code>构造了如下的 DAG，
其中每一个节点表示一个初等函数：</p>

![DAG](/images/torch_autograd_DAG.png)

```.backward()``` 只能对数求导，不能对向量求导。因此，对于向量 $Q$ 的求导需要添加初始梯度 ```Q.backward(gradient = init_gradient)```

back propagate $\rightarrow$ Jacobian matrix $J$ $\times$ vector $\vec{v}$: $J^T · \vec{v}$ by chain rule

computational graph: input data (tensor) & executed operations (elementary operations, Function) in DAG, leaves are input tensors, roots are output tensors

trace DAG from roots to leaves to compute gradient

In a forward pass, autograd does two things simultaneously:

- run the requested operation to compute a resulting tensor
- maintain the operation’s gradient function in the DAG, the .grad_fn attribute of each torch.Tensor is an entry point into this graph

The backward pass kicks off when .backward() is called on the DAG root. autograd then:

- computes the gradients from each .grad_fn
- accumulates them in the respective tensor’s .grad attribute
- using the chain rule, propagates all the way to the leaf tensors

DAGs are dynamic in PyTorch. An important thing to note is that the graph is recreated from scratch; 
after each .backward() call, autograd starts populating a new graph.

torch.autograd tracks operations on all tensors which have requires_grad flag set to True. 
For tensors that don’t require gradients, setting this attribute to False excludes it from the gradient computation DAG.

torch.no_grad(): In this mode, the result of every computation will have requires_grad=False, even when the inputs have requires_grad=True. 
All factory functions, or functions that create a new Tensor and take a requires_grad kwarg, will NOT be affected by this mode.

Function objects (really expressions), which can be apply() ed to compute the result of evaluating the graph

Some operations need intermediary results to be saved during the forward pass in order to execute the backward pass ($x \mapsto x^2$).
When defining a custom Python Function, you can use ```save_for_backward()``` to save tensors during the forward pass and ```saved_tensors to``` retrieve them during the backward pass.

You can explore which tensors are saved by a certain ```grad_fn``` by looking for its attributes starting with the prefix ```_saved``` (```_saved_self / _saved_result```).
To create a custom ```autograd.Function```, subclass this class and implement the ```forward()``` and ```backward()``` static methods. 
Then, to use your custom op in the forward pass, call the class method ```apply()```.

```python
class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result
# Use it by calling the apply method:
output = Exp.apply(input)
```

You can control how saved tensors are packed / unpacked by defining a pair of ```pack_hook``` / ```unpack_hook``` hooks.

The ```pack_hook``` function should take a tensor as its single argument but can return any python object (e.g. another tensor, a tuple, or even a string containing a filename). 
The ```unpack_hook``` function takes as its single argument the output of ```pack_hook``` and should return a tensor to be used in the backward pass. 
The tensor returned by ```unpack_hook``` only needs to have the same content as the tensor passed as input to ```pack_hook```. 

```python
class SelfDeletingTempFile():
    def __init__(self):
        self.name = os.path.join(tmp_dir, str(uuid.uuid4()))

    def __del__(self):
        os.remove(self.name)

def pack_hook(tensor):
    temp_file = SelfDeletingTempFile()
    torch.save(tensor, temp_file.name)
    return temp_file

def unpack_hook(temp_file):
    return torch.load(temp_file.name)
```

the ```unpack_hook``` should not delete the temporary file because it might be called multiple times: 
the temporary file should be alive for as long as the returned ```SelfDeletingTempFile``` object is alive.

register a pair of hooks on a saved tensor by calling the register_hooks() method on a SavedTensor object.

```python
param.grad_fn._raw_saved_self.register_hooks(pack_hook, unpack_hook)
```

use the context-manager saved_tensors_hooks to register a pair of hooks which will be applied to all saved tensors that are created in that context.
The hooks defined with this context manager are thread-local, using those hooks disables all the optimization in place to reduce Tensor object creation.

```python
# Only save on disk tensors that have size >= 1000
SAVE_ON_DISK_THRESHOLD = 1000

def pack_hook(x):
    if x.numel() < SAVE_ON_DISK_THRESHOLD:
        return x
    temp_file = SelfDeletingTempFile()
    torch.save(tensor, temp_file.name)
    return temp_file

def unpack_hook(tensor_or_sctf):
    if isinstance(tensor_or_sctf, torch.Tensor):
        return tensor_or_sctf
    return torch.load(tensor_or_sctf.name)

class Model(nn.Module):
    def forward(self, x):
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
          # ... compute output
          output = x
        return output

model = Model()
net = nn.DataParallel(model)
```

Gradients for non-differentiable functions

1. If the function is differentiable and thus a gradient exists at the current point, use it.
2. If the function is convex (at least locally), use the sub-gradient of minimum norm (it is the steepest descent direction).
3. If the function is concave (at least locally), use the super-gradient of minimum norm (consider -f(x) and apply the previous point).
4. If the function is defined, define the gradient at the current point by continuity (note that inf is possible here, for example for sqrt(0)). If multiple values are possible, pick one arbitrarily.
5. If the function is not defined (sqrt(-1), log(-1) or most functions when the input is NaN, for example) then the value used as the gradient is arbitrary (we might also raise an error but that is not guaranteed). Most functions will use NaN as the gradient, but for performance reasons, some functions will use other values (log(-1), for example).
6. If the function is not a deterministic mapping (i.e. it is not a mathematical function), it will be marked as non-differentiable. This will make it error out in the backward if used on tensors that require grad outside of a no_grad environment.

Locally disabling gradient computation: requires_grad, grad mode, no_grad mode, inference mode

1. ```requires_grad``` is a flag, defaulting to false unless wrapped in a ```nn.Parameter```. 
During the forward pass, an operation is only recorded in the backward graph if at least one of its input tensors require grad. 
During the backward pass (```.backward()```), only leaf tensors with ```requires_grad=True``` will have gradients accumulated into their ```.grad``` fields.
Setting ```requires_grad``` only makes sense for leaf tensors (tensors that do not have a ```grad_fn```, e.g., a ```nn.Module```’s parameters),
all non-leaf tensors will automatically have ```require_grad=True```.
apply `````.requires_grad_(False)````` to the parameters / ```nn.Module```
2. grad mode (default) is the only mode in which ```requires_grad``` takes effect
3. no_grad mode: computations in no-grad mode are never recorded in the backward graph even if there are inputs that have ```requires_grad=True```.
can use the outputs of these computations in grad mode later.
optimizer: when performing the training update you’d like to update parameters in-place without the update being recorded by autograd. 
You also intend to use the updated parameters for computations in grad mode in the next forward pass.
torch.nn.init: rely on no-grad mode when initializing the parameters as to avoid autograd tracking when updating the initialized parameters in-place.
4. inference mode: computations in inference mode are not recorded in the backward graph. 
tensors created in inference mode will not be able to be used in computations to be recorded by autograd after exiting inference mode.
5. evaluation mode(```nn.Moudle.eval()``` equivalently ```module.train(False)```):  ```torch.nn.Dropout``` and ```torch.nn.BatchNorm2d``` that may behave differently depending on training mode

|   Mode    | Excludes operations from being recorded in backward graph | Skips additional autograd tracking overhead | Tensors created while the mode is enabled can be used in grad-mode later |             Examples              |
|:---------:|:---------------------------------------------------------:|:-------------------------------------------:|:------------------------------------------------------------------------:|:---------------------------------:|
|  default  |                             ×                             |                      ×                      |                                    √                                     |           Forward pass            |
|  no-grad  |                             √                             |                      ×                      |                                    √                                     |         Optimizer updates         |
| inference |                             √                             |                      √                      |                                    ×                                     | Data processing, model evaluation |