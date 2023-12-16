---
title: 'pytorch autograd'
date: 23-12-15
permalink: /posts/2023/12/blog-code-pytorch-autograd/
tags:
  - 深度学习基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要介绍了 PyTorch 的 autograd 机制及其具体实现方式。</p>

Torch Autograd
===

<p style="text-align:justify; text-justify:inter-ideograph;">训练一个模型的范式是构造模型 $\mathcal{M}$ 和数据 $\mathcal{D} = \{x_i,y_i\}_{1:N}$，
使用模型的前向过程计算 loss：<code style="color: #B58900">l = M.forward(xi,yi)</code>，然后使用<code style="color: #B58900">l.backward()</code>计算 gradient，
最后使用<code style="color: #B58900">optim.step()</code>更新模型参数。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">那么 PyTorch 是如何计算每个参数的梯度的，即 PyTorch 的自动求导机制(<code style="color: #B58900">torch.autograd</code>)？
通俗而言，<code style="color: #B58900">torch.autograd</code>在模型 forward 的同时构造了一个 computational graph: DAG (由 Function 组成)；
其中的叶子节点表示输入数据和模型参数，而非叶子节点表示模型对这些输入数据和参数进行的初等函数运算(加法，乘法等，elementary operations)。
然后在 backward 的时候触发每个节点的 gradient 计算，并将计算完成的 gradient 存储在各自对应的<code style="color: #B58900">.grad</code>属性内。
最后 optim 的 step 执行时将每个参数的值使用对应<code style="color: #B58900">.grad</code>属性内的 gradient 进行更新计算：<code style="color: #B58900">p = p - lr * p.grad</code>。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">如何构建计算图？首先，torch 构造的计算图是一个有向无环图(DAG)。
其中，叶子节点为输入的 tensor，包括输入的数据和参与计算的模型参数，根节点为输出的 tensor，而中间节点是模型执行的每个初等函数运算(简称为执行操作)。
假设<code style="color: #B58900">a</code>和<code style="color: #B58900">b</code>是<code style="color: #B58900">torch.Tensor</code>变量，
且<code style="color: #B58900">M = lambda x, y: 3*x**3 - y**2</code>，则在<code style="color: #B58900">out=M(a,b)</code>时，<code style="color: #B58900">torch.autograd</code>构造了如下的 DAG，
其中每一个节点表示一个初等函数：</p>

![DAG](/images/torch_autograd_DAG.png)

<p style="text-align:justify; text-justify:inter-ideograph;">其中，<span style="color: blue">蓝色</span>节点表示执行操作节点，在<code style="color: #B58900">torch.autograd</code>中使用<code style="color: #B58900">Function</code>类来实现。
每个初等函数都实现了一个<code style="color: #B58900">Function</code>子类，例如幂函数为<code style="color: #B58900">PowBackward0</code>类。
在<code style="color: #B58900">Function</code>类中，需要实现<code style="color: #B58900">forward</code>和<code style="color: #B58900">backward</code>函数，
其中前者在模型前向运算时使用，而后者在 loss 后向运算时使用。以下为指数函数的<code style="color: #B58900">Function</code>类简易实现：</p>

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
# 使用 Function 函数时，应调用 .apply() 函数代替 .forward() 函数；无法直接调用 .forward() 函数
output = Exp.apply(input)
```

<p style="text-align:justify; text-justify:inter-ideograph;">而且每个执行操作<code style="color: #B58900">Function</code>实例都保存在其输出 tensor 的<code style="color: #B58900">.grad_fn</code>属性上。
因此，PyTorch 中的模型训练范式为 forward 时，输入数据和模型参数，对于每一个执行操作，构建一个对应的<code style="color: #B58900">Function</code>实例，
并调用<code style="color: #B58900">.apply()</code>输出结果作为下一个执行操作的输入数据，并将<code style="color: #B58900">Function</code>实例保存在输出结果的<code style="color: #B58900">.grad_fn</code>属性上。
而在 backward 时，首先使用<code style="color: #B58900">l.grad_fn</code>确定输出节点的<code style="color: #B58900">Function</code>实例，
然后调用其<code style="color: #B58900">.backward()</code>计算对于输入数据的 gradient，并将 gradient 结果传递给下一个节点，
即前一个节点的输入数据所对应的<code style="color: #B58900">Function</code>实例。重复上述操作，直到达到 DAG 的叶子节点，表明 gradient 已计算到输入数据/模型参数。
此时，将最终计算得到的 gradient 保存到其<code style="color: #B58900">.grad</code>属性中。
可以看到，默认情况下，PyTorch 不保存中间计算结果的 gradient (即中间结果的<code style="color: #B58900">.grad</code>属性中为<code style="color: #B58900">None</code>)。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">此外，PyTorch 的求导是通过计算雅可比矩阵(Jacobian matrix) $J$ 和向量 $\vec{v}$ 的乘积来实现链式法则的反向传播(back propagate)，即：</p>

$$J=\left(\begin{array}{ccc}\frac{\partial \mathbf{y}}{\partial x_1} & \cdots & \frac{\partial \mathbf{y}}{\partial x_n}\end{array}\right)=\left(\begin{array}{ccc}\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}\end{array}\right), \vec{v}=\left(\begin{array}{ccc}
\frac{\partial l}{\partial y_1} & \cdots & \frac{\partial l}{\partial y_m}
\end{array}\right)^T \\ J^T \cdot \vec{v}=\left(\begin{array}{ccc}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_1} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_1}{\partial x_n} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{array}\right)\left(\begin{array}{c}
\frac{\partial l}{\partial y_1} \\
\vdots \\
\frac{\partial l}{\partial y_m}
\end{array}\right)=\left(\begin{array}{c}
\frac{\partial l}{\partial x_1} \\
\vdots \\
\frac{\partial l}{\partial x_n}
\end{array}\right)$$

<p style="text-align:justify; text-justify:inter-ideograph;">因此<code style="color: #B58900">.backward()</code>理论上只能对数求导，不能对向量求导，
即 $l.backward()$ 中 $l$ 理论上只能是一个数。为了实现向量 $Q$ 的求导，需要添加与 $Q$ 形状相同的初始梯度<code style="color: #B58900">Q.backward(gradient = init_gradient)</code>。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">同时，PyTorch 的计算图是在每次 forward 时构建的，在 backward 后销毁(只是弃用)；然后在下一次 forward 时再次构建。
这样可以保证每次 forward 时的计算图都是最新的，使得可以针对模型进行任意的改动(比如训练前半段更新全部参数，训练后半段更新指定参数)。</p>

Torch Grad Mode
===

<p style="text-align:justify; text-justify:inter-ideograph;"><code style="color: #B58900">torch.autograd</code> tracks operations on all tensors which have requires_grad flag set to True. 
For tensors that don’t require gradients, setting this attribute to False excludes it from the gradient computation DAG.</p>

<p style="text-align:justify; text-justify:inter-ideograph;"><code style="color: #B58900">torch.no_grad()</code>: In this mode, the result of every computation will have <code style="color: #B58900">requires_grad=False</code>, 
even when the inputs have <code style="color: #B58900">requires_grad=True</code>. 
All factory functions, or functions that create a new Tensor and take a requires_grad kwarg, will NOT be affected by this mode.</p>

<p style="text-align:justify; text-justify:inter-ideograph;">Locally disabling gradient computation: requires_grad, grad mode, no_grad mode, inference mode</p>

1. <p style="text-align:justify; text-justify:inter-ideograph;"><code style="color: #B58900">requires_grad</code> is a flag, defaulting to false unless wrapped in a <code style="color: #B58900">nn.Parameter</code>. 
During the forward pass, an operation is only recorded in the backward graph if at least one of its input tensors require grad. 
During the backward pass (<code style="color: #B58900">.backward()</code>), 
only leaf tensors with <code style="color: #B58900">requires_grad=True</code> will have gradients accumulated into their <code style="color: #B58900">.grad</code> fields.
Setting <code style="color: #B58900">requires_grad</code> only makes sense for leaf tensors (tensors that do not have a <code style="color: #B58900">grad_fn</code>, 
e.g., a <code style="color: #B58900">nn.Module</code>’s parameters),
all non-leaf tensors will automatically have <code style="color: #B58900">require_grad=True</code>.
apply <code style="color: #B58900">.requires_grad_(False)</code> to the parameters / <code style="color: #B58900">nn.Module</code>.</p>

2. <p style="text-align:justify; text-justify:inter-ideograph;">grad mode (default) is the only mode in which <code style="color: #B58900">requires_grad</code> takes effect.</p>

3. <p style="text-align:justify; text-justify:inter-ideograph;">no_grad mode: computations in no-grad mode are never recorded in the backward graph even if there are inputs that have <code style="color: #B58900">requires_grad=True</code>.
can use the outputs of these computations in grad mode later.
optimizer: when performing the training update you’d like to update parameters in-place without the update being recorded by autograd. 
You also intend to use the updated parameters for computations in grad mode in the next forward pass.
torch.nn.init: rely on no-grad mode when initializing the parameters as to avoid autograd tracking when updating the initialized parameters in-place.</p>

4. <p style="text-align:justify; text-justify:inter-ideograph;">inference mode: computations in inference mode are not recorded in the backward graph. 
tensors created in inference mode will not be able to be used in computations to be recorded by autograd after exiting inference mode.</p>

5. <p style="text-align:justify; text-justify:inter-ideograph;">evaluation mode(<code style="color: #B58900">nn.Moudle.eval()</code> equivalently <code style="color: #B58900">module.train(False)</code>): 
<code style="color: #B58900">torch.nn.Dropout</code> and <code style="color: #B58900">torch.nn.BatchNorm2d</code> that may behave differently depending on training mode. </p>

|   Mode    | Excludes operations from being recorded in backward graph | Skips additional autograd tracking overhead | Tensors created while the mode is enabled can be used in grad-mode later |             Examples              |
|:---------:|:---------------------------------------------------------:|:-------------------------------------------:|:------------------------------------------------------------------------:|:---------------------------------:|
|  default  |                             ×                             |                      ×                      |                                    √                                     |           Forward pass            |
|  no-grad  |                             √                             |                      ×                      |                                    √                                     |         Optimizer updates         |
| inference |                             √                             |                      √                      |                                    ×                                     | Data processing, model evaluation |

Appendix
===

## torch.autograd

<p style="text-align:justify; text-justify:inter-ideograph;">computational graph: input data (tensor) & executed operations (elementary operations, Function) in DAG, leaves are input tensors, roots are output tensors.</p>

<p style="text-align:justify; text-justify:inter-ideograph;">In a forward pass, autograd does two things simultaneously:</p>

- <p style="text-align:justify; text-justify:inter-ideograph;">run the requested operation to compute a resulting tensor; </p>

- <p style="text-align:justify; text-justify:inter-ideograph;">maintain the operation’s gradient function in the DAG, the .grad_fn attribute of each torch.Tensor is an entry point into this graph. </p>

<p style="text-align:justify; text-justify:inter-ideograph;">The backward pass kicks off when .backward() is called on the DAG root. autograd then trace DAG from roots to leaves to compute gradient: </p>

- <p style="text-align:justify; text-justify:inter-ideograph;">computes the gradients from each .grad_fn; </p>

- <p style="text-align:justify; text-justify:inter-ideograph;">accumulates them in the respective tensor’s .grad attribute; </p>

- <p style="text-align:justify; text-justify:inter-ideograph;">using the chain rule, propagates all the way to the leaf tensors. </p>

<p style="text-align:justify; text-justify:inter-ideograph;">DAGs are dynamic in PyTorch. An important thing to note is that the graph is recreated from scratch; after each .backward() call, autograd starts populating a new graph.</p>

## torch.autograd.Function

<p style="text-align:justify; text-justify:inter-ideograph;">Function objects (really expressions), which can be apply() ed to compute the result of evaluating the graph. </p>

<p style="text-align:justify; text-justify:inter-ideograph;">Some operations need intermediary results to be saved during the forward pass in order to execute the backward pass ($x \mapsto x^2$).
When defining a custom Python Function, you can use <code style="color: #B58900">save_for_backward()</code> to save tensors during the forward pass and <code style="color: #B58900">saved_tensors to</code>> retrieve them during the backward pass.</p>

<p style="text-align:justify; text-justify:inter-ideograph;">You can explore which tensors are saved by a certain <code style="color: #B58900">grad_fn</code> by looking for its attributes starting with the prefix <code style="color: #B58900">_saved</code> (<code style="color: #B58900">_saved_self / _saved_result</code>).
To create a custom <code style="color: #B58900">autograd.Function</code>>, subclass this class and implement the <code style="color: #B58900">forward()</code> and <code style="color: #B58900">backward()</code> static methods. 
Then, to use your custom op in the forward pass, call the class method <code style="color: #B58900">apply()</code>: </p>

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

<p style="text-align:justify; text-justify:inter-ideograph;">You can control how saved tensors are packed / unpacked by defining a pair of <code style="color: #B58900">pack_hook</code> / <code style="color: #B58900">unpack_hook</code> hooks.</p>

<p style="text-align:justify; text-justify:inter-ideograph;">The <code style="color: #B58900">pack_hook</code> function should take a tensor as its single argument but can return any python object (e.g. another tensor, 
a tuple, or even a string containing a filename). 
The <code style="color: #B58900">unpack_hook</code> function takes as its single argument the output of <code style="color: #B58900">pack_hook</code> and should return a tensor to be used in the backward pass. 
The tensor returned by <code style="color: #B58900">unpack_hook</code> only needs to have the same content as the tensor passed as input to <code style="color: #B58900">pack_hook</code>. </p>

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

<p style="text-align:justify; text-justify:inter-ideograph;">the <code style="color: #B58900">unpack_hook</code> should not delete the temporary file because it might be called multiple times: 
the temporary file should be alive for as long as the returned <code style="color: #B58900">SelfDeletingTempFile</code> object is alive.
register a pair of hooks on a saved tensor by calling the <code style="color: #B58900">register_hooks()</code> method on a SavedTensor object.</p>

```python
param.grad_fn._raw_saved_self.register_hooks(pack_hook, unpack_hook)
```

<p style="text-align:justify; text-justify:inter-ideograph;">use the context-manager saved_tensors_hooks to register a pair of hooks which will be applied to all saved tensors that are created in that context.
The hooks defined with this context manager are thread-local, using those hooks disables all the optimization in place to reduce Tensor object creation.</p>

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

## Computational Graph Implementation

<p style="text-align:justify; text-justify:inter-ideograph;">Gradients for non-differentiable functions: </p>

1. <p style="text-align:justify; text-justify:inter-ideograph;">If the function is differentiable and thus a gradient exists at the current point, use it.</p>
2. <p style="text-align:justify; text-justify:inter-ideograph;">If the function is convex (at least locally), use the sub-gradient of minimum norm (it is the steepest descent direction).</p>
3. <p style="text-align:justify; text-justify:inter-ideograph;">If the function is concave (at least locally), use the super-gradient of minimum norm (consider -f(x) and apply the previous point).</p>
4. <p style="text-align:justify; text-justify:inter-ideograph;">If the function is defined, define the gradient at the current point by continuity (note that inf is possible here, for example for sqrt(0)). If multiple values are possible, pick one arbitrarily.</p>
5. <p style="text-align:justify; text-justify:inter-ideograph;">If the function is not defined (sqrt(-1), log(-1) or most functions when the input is NaN, for example) then the value used as the gradient is arbitrary (we might also raise an error but that is not guaranteed). Most functions will use NaN as the gradient, but for performance reasons, some functions will use other values (log(-1), for example).</p>
6. <p style="text-align:justify; text-justify:inter-ideograph;">If the function is not a deterministic mapping (i.e. it is not a mathematical function), it will be marked as non-differentiable. This will make it error out in the backward if used on tensors that require grad outside of a no_grad environment.</p>

![DAG2](/images/torch_autograd_DAG2.png)

