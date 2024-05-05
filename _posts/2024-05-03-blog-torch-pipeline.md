---
title: 'The Basic Knowledge of Torch Train Pipeline'
date: 24-05-05
update: 24-05-05
permalink: /posts/2024/05/blog-torch-pipeline/
star: superior
tags:
  - 深度学习基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要讲解 PyTorch 训练模型的整个流程的具体细节，
包括如何在前向过程中构建计算图；后向传播过程中如何计算并保存梯度；优化器如何根据梯度更新模型参数。</p>

# Torch 训练的整体流程

<p style="text-align:justify; text-justify:inter-ideograph;">我们以最简单的乘法为例：两个标量 $x_1$ 和 $x_2$ 相乘得到 $v$；然后使用<code style="color: #B58900">v.backward()</code>函数反向计算 $x_1$ 和 $x_2$ 的梯度；最后使用 SGD 优化器更新 $x_1$ 和 $x_2$。代码如下：</p>

![simple torch mul pipeline](/images/simple_torch_pipeline.png)

<p style="text-align:justify; text-justify:inter-ideograph;">接着我们使用<code style="color: #B58900">torchviz</code>的<code style="color: #B58900">make_dot</code>函数获取 PyTorch 构建的计算图：</p>

![simple torch DAG](/images/simple_torch_DAG.png)

<p style="text-align:justify; text-justify:inter-ideograph;">可以看到，计算图的方向与前向计算过程刚好相反。这里，我们将简单描述 Torch 训练的整体流程：在执行乘法过程中，Torc分别为 $x_1$ 和 $x_2$ 构建一个<code style="color: #B58900">AccumulateGrad</code>节点，并将 $x_1$ / $x_2$ 存储在对应的<code style="color: #B58900">AccumulateGrad</code>节点的<code style="color: #B58900">variable</code>属性中；然后根据<code style="color: #B58900">*</code>的乘法操作为 $v$ 构建一个<code style="color: #B58900">MulBackwrad0</code>节点，并将其存储在 $v$ 的<code style="color: #B58900">grad_fn</code>属性中。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">而在后向传播计算梯度的过程中，执行<code style="color: #B58900">v.backward()</code>函数时，Torch 首先会获取到存储在 $v$ 的<code style="color: #B58900">grad_fn</code>属性中<code style="color: #B58900">MulBackwrad0</code>节点，然后将初始梯度<code style="color: #B58900">gradient</code>作为输入传递给其<code style="color: #B58900">.backward()</code>函数计算该节点的输入的梯度，即 $x_1$ 和 $x_2$ 的梯度；接着将 $x_1$ 和 $x_2$ 的梯度作为输入传递给各自对应的<code style="color: #B58900">AccumulateGrad</code>节点的<code style="color: #B58900">.backward()</code>函数实现将梯度累加到 $x_1$ 和 $x_2$ 的<code style="color: #B58900">.grad</code>属性中。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">在 SGD 优化器更新 $x_1$ 和 $x_2$ 的过程中，SGD 的<code style="color: #B58900">step()</code>函数遍历初始化时的<code style="color: #B58900">params</code>参数，判断其<code style="color: #B58900">required_grad</code>属性是否为<code style="color: #B58900">True</code>，若为<code style="color: #B58900">True</code>，则取出其<code style="color: #B58900">data</code>属性和<code style="color: #B58900">grad</code>属性，将<code style="color: #B58900">data</code>减去<code style="color: #B58900">grad</code>，得到更新后的参数<code style="color: #B58900">params</code>。</p>

# 前向过程构建计算图

<p style="text-align:justify; text-justify:inter-ideograph;">介绍基本知识：Node $\rightarrow$ Edge $\rightarrow$ MulBackward0</p>

<p style="text-align:justify; text-justify:inter-ideograph;">叙述过程：Tensor.mul $\rightarrow$ torch._C._TensorBase.__mul__ $\rightarrow$ mul_Tensor $\rightarrow$ collect_next_edges $\rightarrow$ gradient_edge $\rightarrow$ set_next_edges $\rightarrow$ set_history $\rightarrow$ set_gradient_edge</p>

<p style="text-align:justify; text-justify:inter-ideograph;">不知道你们有没有这样的疑惑：在我们的代码中，只是简单的编写了两个<code style="color: #B58900">tensor</code>的矩阵相乘：<code style="color: #B58900">tensor = tensor1 @ tensor2</code>；而 PyTorch 便自动为我们构建了一个计算图（可以看到<code style="color: #B58900">tensor</code>的<code style="color: #B58900">.grad_fn</code>属性为<code style="color: #B58900">MulBackward0</code>；如果<code style="color: #B58900">tensor1 / tensor2</code>的<code style="color: #B58900">.required_grad</code>属性为<code style="color: #B58900">True</code>）。这是如何实现的？实际上，PyTorch 在<code style="color: #B58900">Tensor</code>类中实现了对每个初等函数的<b>重载</b>。例如对于<code style="color: #B58900">mul</code>操作，<code style="color: #B58900">Tensor</code>类内的重载实现为：</p>

![tensor mul operation](/images/tensor_mul.png)

<p style="text-align:justify; text-justify:inter-ideograph;">可以看到，其内部实现是使用 C++ 语言来编写的，继续追溯到 C++ 源代码中，可以看到<code style="color: #B58900">mul</code>操作的具体实现为：</p>

![tensor mul operation in C++](/images/tensor_mul_c++.png)

<p style="text-align:justify; text-justify:inter-ideograph;">其中，<code style="color: #B58900">self, other</code>分别是<code style="color: #B58900">mul</code>操作的第一个<code style="color: #B58900">tensor</code>和第二个<code style="color: #B58900">tensor</code>。首先，<code style="color: #B58900">compute_requires_grad()</code>函数判断<code style="color: #B58900">self/other</code>的<code style="color: #B58900">.required_grad</code>属性是否为<code style="color: #B58900">True</code>，只要有一个为<code style="color: #B58900">True</code>，则<code style="color: #B58900">_any_requires_grad</code>为<code style="color: #B58900">True</code>，表示<code style="color: #B58900">mul</code>操作生成的输出的<code style="color: #B58900">required_grad</code>也为<code style="color: #B58900">True</code>。其次，创建一个<code style="color: #B58900">grad_fn</code>：<code style="color: #B58900">MulBackward0</code>作为下一个节点，</p>

<p style="text-align:justify; text-justify:inter-ideograph;">

1. optimizer 中的 self.param_groups 和 self.states 的 keys 都是与 model.parameters() 共享内存空间，即它们都指向同一个内存区域

2. dict 的 keys(), values() 和 items() 的返回值与 dict 共享内存空间，对其值进行“原地”操作会同步修改 dict 内的值

3. torch.autograd 不保存中间变量 (即对于 z = (x + y) ** 2，torch 不使用一个额外的变量保持 x + y 的值)

4. torch.autograd.funtions.Function 的重要属性：

-------------------------------------
_save_self / _save_other 一般是为了后向过程时计算梯度而保持的必要输入
_save_self
_save_other
-------------------------------------
variable -> 只在 AccumulateGrad 中出现


<p style="text-align:justify; text-justify:inter-ideograph;"><code style="color: #B58900">torch.autograd</code>理论上需要可微函数才能计算梯度，但是并不是所有的函数在其定义域内都是可微的，例如 $ReLU$ 在 $x=0$ 时不可微。
为此，PyTorch 使用如下的优先级来计算不可微函数的梯度: </p>

1. <p style="text-align:justify; text-justify:inter-ideograph;">If the function is differentiable and thus a gradient exists at the current point, use it.</p>
2. <p style="text-align:justify; text-justify:inter-ideograph;">If the function is convex (at least locally), use the sub-gradient of minimum norm (it is the steepest descent direction).</p>
3. <p style="text-align:justify; text-justify:inter-ideograph;">If the function is concave (at least locally), use the super-gradient of minimum norm (consider -f(x) and apply the previous point).</p>
4. <p style="text-align:justify; text-justify:inter-ideograph;">If the function is defined, define the gradient at the current point by continuity (note that inf is possible here, for example for sqrt(0)). If multiple values are possible, pick one arbitrarily.</p>
5. <p style="text-align:justify; text-justify:inter-ideograph;">If the function is not defined (sqrt(-1), log(-1) or most functions when the input is NaN, for example) then the value used as the gradient is arbitrary (we might also raise an error but that is not guaranteed). Most functions will use NaN as the gradient, but for performance reasons, some functions will use other values (log(-1), for example).</p>
6. <p style="text-align:justify; text-justify:inter-ideograph;">If the function is not a deterministic mapping (i.e. it is not a mathematical function), it will be marked as non-differentiable. This will make it error out in the backward if used on tensors that require grad outside of a no_grad environment.</p>

Torch Grad Mode
===

<p style="text-align:justify; text-justify:inter-ideograph;"><code style="color: #B58900">torch.autograd</code> tracks operations on all tensors which have <code style="color: #B58900">requires_grad</code> flag set to True. 
For tensors that don’t require gradients, setting this attribute to False excludes it from the gradient computation DAG.</p>

<p style="text-align:justify; text-justify:inter-ideograph;"><code style="color: #B58900">torch.no_grad()</code>: In this mode, the result of every computation will have <code style="color: #B58900">requires_grad=False</code>, 
even when the inputs have <code style="color: #B58900">requires_grad=True</code>. 
All factory functions, or functions that create a new Tensor and take a <code style="color: #B58900">requires_grad</code> kwarg, will <b>NOT</b> be affected by this mode.</p>

<p style="text-align:justify; text-justify:inter-ideograph;">Locally disabling gradient computation: requires_grad, grad mode, no_grad mode, inference mode:</p>

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

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

Appendix
===

## torch.autograd

<p style="text-align:justify; text-justify:inter-ideograph;">computational graph: input data (tensor) & executed operations (elementary operations, Function) in DAG, leaves are input tensors, roots are output tensors.</p>

<p style="text-align:justify; text-justify:inter-ideograph;">In a forward pass, autograd does two things simultaneously:</p>

- <p style="text-align:justify; text-justify:inter-ideograph;">run the requested operation to compute a resulting tensor; </p>

- <p style="text-align:justify; text-justify:inter-ideograph;">maintain the operation’s gradient function in the DAG, the .grad_fn attribute of each torch.Tensor is an entry point into this graph. </p>

<p style="text-align:justify; text-justify:inter-ideograph;">The backward pass kicks off when <code style="color: #B58900">.backward()</code> is called on the DAG root. autograd then trace DAG from roots to leaves to compute gradient: </p>

- <p style="text-align:justify; text-justify:inter-ideograph;">computes the gradients from each <code style="color: #B58900">.grad_fn</code>; </p>

- <p style="text-align:justify; text-justify:inter-ideograph;">accumulates them in the respective tensor’s <code style="color: #B58900">.grad</code> attribute; </p>

- <p style="text-align:justify; text-justify:inter-ideograph;">using the chain rule, propagates all the way to the leaf tensors. </p>

<p style="text-align:justify; text-justify:inter-ideograph;">DAGs are dynamic in PyTorch. An important thing to note is that the graph is recreated from scratch; after each <code style="color: #B58900">.backward()</code> call, 
autograd starts populating a new graph.</p>

## torch.autograd.Function

<p style="text-align:justify; text-justify:inter-ideograph;">Function objects (really expressions), which can be <code style="color: #B58900">apply()</code> ed to compute the result of evaluating the graph. </p>

<p style="text-align:justify; text-justify:inter-ideograph;">Some operations need intermediary results to be saved during the forward pass in order to execute the backward pass ($x \mapsto x^2$).
When defining a custom Python Function, you can use <code style="color: #B58900">save_for_backward()</code> to save tensors during the forward pass and <code style="color: #B58900">saved_tensors to</code> retrieve them during the backward pass.</p>

<p style="text-align:justify; text-justify:inter-ideograph;">You can explore which tensors are saved by a certain <code style="color: #B58900">grad_fn</code> by looking for its attributes starting with the prefix <code style="color: #B58900">_saved</code> (<code style="color: #B58900">_saved_self</code> / <code style="color: #B58900">_saved_result</code>).
To create a custom <code style="color: #B58900">autograd.Function</code>, subclass this class and implement the <code style="color: #B58900">forward()</code> and <code style="color: #B58900">backward()</code> static methods. 
Then, to use your custom op in the forward pass, call the class method <code style="color: #B58900">apply()</code>: </p>

![exp Function](/images/torch_autograd_Function.png)

<p style="text-align:justify; text-justify:inter-ideograph;">You can control how saved tensors are packed / unpacked by defining a pair of <code style="color: #B58900">pack_hook</code> / <code style="color: #B58900">unpack_hook</code> hooks.</p>

<p style="text-align:justify; text-justify:inter-ideograph;">The <code style="color: #B58900">pack_hook</code> function should take a tensor as its single argument but can return any python object (e.g. another tensor, 
a tuple, or even a string containing a filename). 
The <code style="color: #B58900">unpack_hook</code> function takes as its single argument the output of <code style="color: #B58900">pack_hook</code> and should return a tensor to be used in the backward pass. 
The tensor returned by <code style="color: #B58900">unpack_hook</code> only needs to have the same content as the tensor passed as input to <code style="color: #B58900">pack_hook</code>. </p>

![pack / unpack](/images/torch_autograd_pack.png)

<p style="text-align:justify; text-justify:inter-ideograph;">the <code style="color: #B58900">unpack_hook</code> should not delete the temporary file because it might be called multiple times: 
the temporary file should be alive for as long as the returned <code style="color: #B58900">SelfDeletingTempFile</code> object is alive.
register a pair of hooks on a saved tensor by calling the <code style="color: #B58900">register_hooks()</code> method on a SavedTensor object.</p>

<p style="text-align:justify; text-justify:inter-ideograph;"><code style="color: #B58900">
param.grad_fn._raw_saved_self.register_hooks(pack_hook, unpack_hook)
</code></p>

<p style="text-align:justify; text-justify:inter-ideograph;">use the context-manager <code style="color: #B58900">saved_tensors_hooks</code> to register a pair of hooks which will be applied to all saved tensors that are created in that context.
The hooks defined with this context manager are thread-local, using those hooks disables all the optimization in place to reduce Tensor object creation.</p>

![torch pack](/images/torch_autograd_pack_DDP.png)


# Reference

1. [Understanding PyTorch with an example: a step-by-step tutorial](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)

2. [The SGD source code of PyTorch](https://github.com/pytorch/pytorch/blob/cd9b27231b51633e76e28b6a34002ab83b0660fc/torch/optim/sgd.py#L63)

3. [A lightweight package to create visualizations of PyTorch execution graphs and traces](https://github.com/szagoruyko/pytorchviz)

4. [How Computational Graphs are Constructed in PyTorch](https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/)

5. [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/)

6. [Ultimate guide to PyTorch Optimizers](https://analyticsindiamag.com/ultimate-guide-to-pytorch-optimizers/)

7. [torch.optim](https://pytorch.org/docs/stable/optim.html)

8. [What is a PyTorch optimizer?](https://www.educative.io/answers/what-is-a-pytorch-optimizer)