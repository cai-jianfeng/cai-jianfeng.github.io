---
title: 'The Basic Knowledge of Torch Train Pipeline'
date: 24-05-05
update: 24-05-14
permalink: /posts/2024/05/blog-torch-pipeline/
star: superior
tags:
  - 深度学习基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要讲解 PyTorch 训练模型的整个流程的具体细节，
包括如何在前向过程中构建计算图；后向传播过程中如何计算并保存梯度；优化器如何根据梯度更新模型参数。(建议先阅读我之前关于 torch.autograd 的博客 <a href="https://cai-jianfeng.github.io/posts/2023/12/blog-code-pytorch-autograd/" target="_blank">The Basic Knowledge of PyTorch Autograd</a> )</p>

# Torch 训练的整体流程

<p style="text-align:justify; text-justify:inter-ideograph;">我们以最简单的乘法为例：两个标量 $x_1$ 和 $x_2$ 相乘得到 $v$；然后使用<code style="color: #B58900">v.backward()</code>函数反向计算 $x_1$ 和 $x_2$ 的梯度；最后使用 SGD 优化器更新 $x_1$ 和 $x_2$。代码如下：</p>

![simple torch mul pipeline](/images/simple_torch_pipeline.png)

<p style="text-align:justify; text-justify:inter-ideograph;">接着我们使用<code style="color: #B58900">torchviz</code>的<code style="color: #B58900">make_dot</code>函数获取 PyTorch 构建的计算图：</p>

![simple torch DAG](/images/simple_torch_DAG.png)

<p style="text-align:justify; text-justify:inter-ideograph;">可以看到，计算图的方向与前向计算过程刚好相反。这里，我们将简单描述 Torch 训练的整体流程：在执行乘法过程中，Torch 分别为 $x_1$ 和 $x_2$ 构建一个<code style="color: #B58900">AccumulateGrad</code>节点，并将 $x_1$ / $x_2$ 存储在对应的<code style="color: #B58900">AccumulateGrad</code>节点的<code style="color: #B58900">variable</code>属性中；然后根据<code style="color: #B58900">*</code>的乘法操作为 $v$ 构建一个<code style="color: #B58900">MulBackwrad0</code>节点，并将其存储在 $v$ 的<code style="color: #B58900">grad_fn</code>属性中。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">而在后向传播计算梯度的过程中，执行<code style="color: #B58900">v.backward()</code>函数时，Torch 首先会获取到存储在 $v$ 的<code style="color: #B58900">grad_fn</code>属性中<code style="color: #B58900">MulBackwrad0</code>节点，然后将初始梯度<code style="color: #B58900">gradient</code>作为输入传递给其<code style="color: #B58900">.backward()</code>函数计算该节点的输入的梯度，即 $x_1$ 和 $x_2$ 的梯度；接着将 $x_1$ 和 $x_2$ 的梯度作为输入传递给各自对应的<code style="color: #B58900">AccumulateGrad</code>节点的<code style="color: #B58900">.backward()</code>函数实现将梯度累加到 $x_1$ 和 $x_2$ 的<code style="color: #B58900">.grad</code>属性中。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">在 SGD 优化器更新 $x_1$ 和 $x_2$ 的过程中，SGD 的<code style="color: #B58900">step()</code>函数遍历初始化时传入的<code style="color: #B58900">params</code>参数，判断其<code style="color: #B58900">required_grad</code>属性是否为<code style="color: #B58900">True</code>，若为<code style="color: #B58900">True</code>，则取出其<code style="color: #B58900">data</code>属性和<code style="color: #B58900">grad</code>属性，将<code style="color: #B58900">data</code>减去<code style="color: #B58900">grad</code>，得到更新后的参数<code style="color: #B58900">params</code>。</p>

# 前向过程构建计算图

<p style="text-align:justify; text-justify:inter-ideograph;">介绍基本知识：Node $\rightarrow$ Edge $\rightarrow$ MulBackward0</p>

<p style="text-align:justify; text-justify:inter-ideograph;">叙述过程：Tensor.mul $\rightarrow$ torch._C._TensorBase.__mul__ $\rightarrow$ mul_Tensor $\rightarrow$ collect_next_edges $\rightarrow$ gradient_edge $\rightarrow$ set_next_edges $\rightarrow$ set_history $\rightarrow$ set_gradient_edge</p>

<p style="text-align:justify; text-justify:inter-ideograph;">不知道你们有没有这样的疑惑：在我们的代码中，只是简单的编写了两个<code style="color: #B58900">tensor</code>的矩阵相乘：<code style="color: #B58900">tensor = tensor1 * tensor2</code>；而 PyTorch 便自动为我们构建了一个计算图(可以看到<code style="color: #B58900">tensor</code>的<code style="color: #B58900">.grad_fn</code>属性为<code style="color: #B58900">MulBackward0</code>；如果<code style="color: #B58900">tensor1 / tensor2</code>的<code style="color: #B58900">.required_grad</code>属性为<code style="color: #B58900">True</code>)。这是如何实现的？虽然我们在前面的博客 <a href="https://cai-jianfeng.github.io/posts/2023/12/blog-code-pytorch-autograd/" target="_blank">The Basic Knowledge of PyTorch Autograd</a> 中讲了关于 PyTorch 自动求导的过程，知道了每个节点是在初等函数执行时立即创建的，但并没有涉及到具体的代码对应过程(即在<code style="color: #B58900">tensor = tensor1 * tensor2</code>背后究竟是哪些代码实现了计算图的创建)。实际上，PyTorch 在<code style="color: #B58900">Tensor</code>类中实现了对每个初等函数的<b>重载</b>，使得每个初等函数操作并不只是简单的实现初等函数而已。例如对于<code style="color: #B58900">mul</code>操作，<code style="color: #B58900">Tensor</code>类内的重载实现为：</p>

![tensor mul operation](/images/tensor_mul.png)

<p style="text-align:justify; text-justify:inter-ideograph;">可以看到，其内部实现是使用 C++ 语言来编写的，继续追溯到 C++ 源代码中，可以看到<code style="color: #B58900">mul</code>操作的具体实现为：</p>

![tensor mul operation in C++](/images/tensor_mul_c++.png)

<p style="text-align:justify; text-justify:inter-ideograph;">这个代码有点吓人，让我们一步步来。其中，<code style="color: #B58900">self, other</code>分别是<code style="color: #B58900">mul</code>操作的第一个<code style="color: #B58900">tensor</code>和第二个<code style="color: #B58900">tensor</code>。首先，第 $4$ 行代码的<code style="color: #B58900">compute_requires_grad()</code>函数判断<code style="color: #B58900">self/other</code>的<code style="color: #B58900">required_grad</code>属性是否为<code style="color: #B58900">True</code>，只要有一个为<code style="color: #B58900">True</code>，则<code style="color: #B58900">_any_requires_grad</code>为<code style="color: #B58900">True</code>，表示此时的<code style="color: #B58900">mul</code>操作需要生成节点，同时其生成的输出的<code style="color: #B58900">required_grad</code>也为<code style="color: #B58900">True</code>。在得到<code style="color: #B58900">_any_requires_grad</code>为<code style="color: #B58900">True</code>后(第 $6$ 行代码)，代码会创建一个<code style="color: #B58900">MulBackward0</code>作为该<code style="color: #B58900">mul</code>操作在计算图上的节点(第 $8$ 行代码)，同时将其赋值给<code style="color: #B58900">grad_fn</code>；而<code style="color: #B58900">set_next_edges()</code>则是设置当前的<code style="color: #B58900">MulBackward0</code>节点与之前操作生成的节点的连接。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">接下来，让我们继续深入每个部分。首先，<code style="color: #B58900">self/other</code>是一个<code style="color: #B58900">Tensor</code>，当设置其<code style="color: #B58900">required_grad</code>的属性为 True 时，会执行下面的<code style="color: #B58900">set_requires_grad()</code>函数：</p>

![set requires grad](/images/required_grad_set.png)

<p style="text-align:justify; text-justify:inter-ideograph;">其会为<code style="color: #B58900">self/other</code>创建一个新的属性<code style="color: #B58900">autograd_meta_</code>(<code style="color: #B58900">AutogradMeta</code>类)，该属性用于存储<code style="color: #B58900">self/other</code>的梯度(<code style="color: #B58900">grad_</code>)和节点(<code style="color: #B58900">grad_fn_</code>)等)，对应于 Python 代码里的<code style="color: #B58900">.grad</code>和<code style="color: #B58900">grad_fn</code>属性。(当然其还有梯度累加器(<code style="color: #B58900">grad_accumulator_</code>用于累加多个父节点传递的梯度)</p>

<p style="text-align:justify; text-justify:inter-ideograph;">其次，计算图的每个节点的类型均为<code style="color: #B58900">Node</code>结构体(对应于 Python 代码中的<code style="color: #B58900">Function</code>类)。下图是<code style="color: #B58900">Node</code>结构体的具体内容：</p>

![Node structure](/images/Node_class_c++.png)

<p style="text-align:justify; text-justify:inter-ideograph;">其中，<code style="color: #B58900">operator()</code>和<code style="color: #B58900">apply()</code>分别是节点的前向和反向计算函数(对应于 Python 代码中的<code style="color: #B58900">forward()</code>和<code style="color: #B58900">backward()</code>函数)，不同的节点可以重写它们以实现不同的计算过程。而<code style="color: #B58900">next_edges_</code>则是存储节点所连接的前向节点(对应于 Python 代码中的<code style="color: #B58900">next_functions</code>)。因此，<code style="color: #B58900">next_edges_</code>中的每条边都是<code style="color: #B58900">Edge</code>结构体，结构体中存储执行前向节点的指针。下图是<code style="color: #B58900">Edge</code>结构体的具体内容：</p>

![edge structure](/images/edge_class_c++.png)

<p style="text-align:justify; text-justify:inter-ideograph;">所以，<code style="color: #B58900">mul_Tensor()</code>函数中的<code style="color: #B58900">MulBackward0</code>操作即是<code style="color: #B58900">Node</code>结构体的子结构体，其主要重写了<code style="color: #B58900">apply()</code>方法用于计算<code style="color: #B58900">mul</code>操作的反向过程(其没有重写<code style="color: #B58900">operator()</code>方法，因为<code style="color: #B58900">mul</code>操作的前向过程在<code style="color: #B58900">mul_Tensor()</code>函数中实现)，如下图所示：</p>

![mulbackward structure](/images/multibackward0_c++.png)

<p style="text-align:justify; text-justify:inter-ideograph;">了解了各个变量的基本结构后，我们回到<code style="color: #B58900">mul_Tensor()</code>函数中。可以猜到，<code style="color: #B58900">set_next_edges()</code>应该是要将之前操作生成的节点赋值到当前的<code style="color: #B58900">MulBackward0</code>节点的<code style="color: #B58900">next_edges_</code>中。首先需要获取之前操作生成的节点，通过<code style="color: #B58900">collect_next_edges()</code>函数实现。如下图所示：</p>

![collect next edges](/images/collect_next_edges_function_c++.png)

<p style="text-align:justify; text-justify:inter-ideograph;">这个代码更吓人，还是让我们一步步来！首先，<code style="color: #B58900">collect_next_edges()</code>函数是通过输入<code style="color: #B58900">mul</code>操作的输入数据，即<code style="color: #B58900">self, other</code>；然后创建<code style="color: #B58900">MakeNextFunctionList</code>结构体的实例<code style="color: #B58900">make</code>，并调用其<code style="color: #B58900">apply()</code>方法(即<code style="color: #B58900">MakeNextFunctionList</code>的<code style="color: #B58900">operator()</code>方法)实现的获取之前操作生成的节点。而<code style="color: #B58900">MakeNextFunctionList</code>的<code style="color: #B58900">operator()</code>方法同样输入<code style="color: #B58900">mul</code>操作的输入数据，然后构建<code style="color: #B58900">next_edges</code>数组，接着通过调用<code style="color: #B58900">gradient_edge()</code>方法获取每个输入数据里保存的之前操作生成的节点(使用<code style="color: #B58900">Edge</code>结构体包装)，并将其存储在<code style="color: #B58900">next_edges</code>数组中，最后将<code style="color: #B58900">next_edges</code>数组返回给<code style="color: #B58900">collect_next_edges()</code>函数。而<code style="color: #B58900">gradient_edge()</code>方法输入<code style="color: #B58900">mul</code>操作的输入数据，判断其是否保存的之前操作生成的节点<code style="color: #B58900">gradient = self.grad_fn()</code>：若有，则说明该输入数据属于中间数据，则将其包装成<code style="color: #B58900">Edge</code>结构体后返回；若没有，则说明该输入数据属于最原始的输入数据，则将其保存的节点设置为<code style="color: #B58900">AccumulateBackward</code>节点(通过调用<code style="color: #B58900">grad_accumulator()</code>函数获得)，并其包装成<code style="color: #B58900">Edge</code>结构体后返回（这就是为什么每个叶子节点前都有一个<code style="color: #B58900">AccumulateBackward</code>节点的原因）。从<code style="color: #B58900">gradient_edge()</code>方法返回到<code style="color: #B58900">MakeNextFunctionList</code>的<code style="color: #B58900">operator()</code>方法，再返回到<code style="color: #B58900">collect_next_edges()</code>函数，即可得到当前的<code style="color: #B58900">MulBackward0</code>节点的之前操作生成的节点。然后通过<code style="color: #B58900">set_next_edges()</code>将其赋值到<code style="color: #B58900">next_edges_</code>中。如下图所示：</p>

![set next edges](/images/set_next_edges_c++.png)

<p style="text-align:justify; text-justify:inter-ideograph;">完成了在<code style="color: #B58900">set_next_edges()</code>后，接下来便需要计算前向过程(对应于 Python 代码中的<code style="color: #B58900">Function</code>类的<code style="color: #B58900">forward()</code>方法)，获得计算结果<code style="color: #B58900">result</code>(<code style="color: #B58900">mul_Tensor()</code>函数中的第 $15 \sim 20$ 行)。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">最后，需要将生成的<code style="color: #B58900">MulBackward0</code>节点保存到输出<code style="color: #B58900">result</code>中(对应 Python 代码的<code style="color: #B58900">outputs.grad_fn = now_fn</code>)，其通过<code style="color: #B58900">set_history()</code>函数实现。如下图所示：</p>

![set_history_c++](/images/set_history_c++.png)

<p style="text-align:justify; text-justify:inter-ideograph;">首先，<code style="color: #B58900">set_history()</code>函数是通过输入前向过程的输出<code style="color: #B58900">result</code>和生成的<code style="color: #B58900">MulBackward0</code>节点，然后调用<code style="color: #B58900">set_gradient_edge()</code>方法实现的将<code style="color: #B58900">MulBackward0</code>节点保存在输出<code style="color: #B58900">result</code>的<code style="color: #B58900">AutogradMeta</code>属性的<code style="color: #B58900">grad_fn_</code>中。而<code style="color: #B58900">set_gradient_edge()</code>方法则是通过输入同样的前向过程的输出<code style="color: #B58900">result</code>和生成的<code style="color: #B58900">MulBackward0</code>节点，取出<code style="color: #B58900">result</code>的<code style="color: #B58900">AutogradMeta</code>属性<code style="color: #B58900">meta</code>，将<code style="color: #B58900">MulBackward0</code>赋值在其<code style="color: #B58900">grad_fn_</code>属性中。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">至此，我们终于“稍微”搞懂了 PyTorch 自动化构建计算图的过程。原来在我们写了一个简单的<code style="color: #B58900">tensor = tensor1 * tensor2</code>代码背后，PyTorch 执行了如此多的额外代码操作来实现计算图的构建。</p>

# 后向传播过程计算并保存梯度

<p style="text-align:justify; text-justify:inter-ideograph;">敬请期待！</p>

# 优化器根据梯度更新模型参数

<p style="text-align:justify; text-justify:inter-ideograph;">不同于前向过程和后向过程，其代码需要深入到底层的 C++ 源代码进行理解，优化器利用计算得到的梯度更新模型参数的过程主要在 Python 源代码中实现。现在让我们以最简单的 SGD 优化器为例：首先我们需要初始化一个 SGD 优化器实例，它至少需要输入两个参数（模型参数<code style="color: #B58900">params</code>(即 $x_1$ 和 $x_2$)和初始学习率<code style="color: #B58900">lr</code>），如下图所示：</p>

![SGD optimizer](/images/SGD_optimizer_construction_py.png)

<p style="text-align:justify; text-justify:inter-ideograph;">在经过前向过程(<code style="color: #B58900">v = x1 * x2</code>)和后向过程(<code style="color: #B58900">v.backward()</code>)后，此时 $x_1$ 和 $x_2$ 的<code style="color: #B58900">grad</code>属性内已经存储了计算得到的梯度。因此，我们能想到的最直接的做法就是遍历<code style="color: #B58900">params</code>的每一个参数，判断每个参数的<code style="color: #B58900">required_grad</code>属性是否为<code style="color: #B58900">True</code>；若是，则取出其对应的<code style="color: #B58900">grad</code>属性内存储的梯度，并将该参数与其梯度(乘以学习率)进行相减即可实现参数更新。因此 SGD 类的简单实现应该如下图所示：</p>

![SDG class](/images/SGD_class_py.png)

<p style="text-align:justify; text-justify:inter-ideograph;">但是这里有个问题，前面我们说过，PyTorch 重载了<code style="color: #B58900">Tensor</code>类的所有初等函数操作；因此，当我们执行<code style="color: #B58900">param -= grad * self.lr</code>操作时，我们实际上会在原有计算图的基础上再构建一个<code style="color: #B58900">SubBackward0</code>节点分支，如下图所示：</p>

![simple SGD problem](/images/simple_SGD_problem.png)

<p style="text-align:justify; text-justify:inter-ideograph;">因此，为了不让 PyTorch 继续构建计算图，我们需要设置<code style="color: #B58900">with torch.no_grad()</code>来“告诉” PyTorch 下面的操作不需要构建计算图，此时<code style="color: #B58900">Tensor</code>类的所有初等函数操作就不会构建计算图。因此，改进的SGD 类代码如下图所示：</p>

![advance SGD class](/images/SGD_class_advance_py.png)

<p style="text-align:justify; text-justify:inter-ideograph;">而在 SGD 的源代码中，PyTorch 使用另一种方式来避免计算图的构建，通过使用<code style="color: #B58900">torch._dynamo.graph_break()</code>实现计算图的脱离来确保初等函数操作就不会继续构建计算图。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">了解了如何简单实现 SGD 后，接下来让我们进入 SGD 的源代码来验证我们的实现是否正确。首先是 SGD 如何保存输入进来的<code style="color: #B58900">params</code>参数，下图为 SGD 的<code style="color: #B58900">__init__()</code>函数部分代码：</p>

![SGD source code init](/images/SGD_source_code_init.png)

<p style="text-align:justify; text-justify:inter-ideograph;">可以看到，SGD 是通过调用其父类<code style="color: #B58900">Optimizer</code>的<code style="color: #B58900">__init__()</code>函数将输入进来的参数保存在<code style="color: #B58900">self.param_groups</code>列表内。接下来就是 SGD 的<code style="color: #B58900">step()</code>函数，下图为 SGD 的<code style="color: #B58900">step()</code>函数部分代码：</p>

![SGD source code step](/images/SGD_source_code_step.png)

<p style="text-align:justify; text-justify:inter-ideograph;">首先，<code style="color: #B58900">step()</code>函数对每个<code style="color: #B58900">self.param_groups</code>列表内的每个参数组<code style="color: #B58900">group</code>，调用<code style="color: #B58900">self._init_group()</code>判断其每个参数<code style="color: #B58900">p</code>的<code style="color: #B58900">grad</code>属性是否为<code style="color: #B58900">None</code>：如果不是，则表示需要更新该参数，则将其存储在<code style="color: #B58900">params_with_grad</code>列表中，同时使用<code style="color: #B58900">d_p_list</code>列表存储其对应的梯度<code style="color: #B58900">p.grad</code>。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">接着，对于那些需要更新的参数<code style="color: #B58900">params_with_grad</code>，调用<code style="color: #B58900">sgd()</code>函数进行参数更新。在<code style="color: #B58900">sgd()</code>函数中，进行一系列的检查后，调用<code style="color: #B58900">_single_tensor_sgd()</code>函数进行参数更新。
而<code style="color: #B58900">_single_tensor_sgd()</code>函数则是遍历<code style="color: #B58900">params_with_grad</code>列表中的所有参数，对于每个参数<code style="color: #B58900">param</code>列表，取出其在<code style="color: #B58900">d_p_list</code>列表中的对应的梯度<code style="color: #B58900">d_p</code>，并使用<b>原地更新</b>的方式进行参数更新：<code style="color: #B58900">param.add_(d_p, alpha=-lr)</code>。由于是原地更新，且传入优化器的参照即为模型参数，因此对应的模型中的参数也会同步进行更新。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">至此，我们终于完成了 PyTorch 训练模型的整个流程的具体细节（深入到底层代码），包括如何在前向过程中构建计算图；<text style="background-color:gray; color:white">后向传播过程中如何计算并保存梯度；</text>优化器如何根据梯度更新模型参数。（<text style="background-color:gray; color:white">灰底白字</text>部分表示尚未完成部分🤪）</p>

<!-- 1. optimizer 中的 self.param_groups 和 self.states 的 keys 都是与 model.parameters() 共享内存空间，即它们都指向同一个内存区域

1. dict 的 keys(), values() 和 items() 的返回值与 dict 共享内存空间，对其值进行“原地”操作会同步修改 dict 内的值

2. torch.autograd 不保存中间变量 (即对于 z = (x + y) ** 2，torch 不使用一个额外的变量保持 x + y 的值)

3. torch.autograd.funtions.Function 的重要属性：

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

![torch pack](/images/torch_autograd_pack_DDP.png) -->


# Reference

1. [Understanding PyTorch with an example: a step-by-step tutorial](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)

2. [The SGD source code of PyTorch](https://github.com/pytorch/pytorch/blob/cd9b27231b51633e76e28b6a34002ab83b0660fc/torch/optim/sgd.py#L63)

3. [A lightweight package to create visualizations of PyTorch execution graphs and traces](https://github.com/szagoruyko/pytorchviz)

4. [Overview of PyTorch Autograd Engine](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/)

5. [How Computational Graphs are Constructed in PyTorch](https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/)

6. [How Computational Graphs are Executed in PyTorch](https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/)

7. [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/)

8. [Ultimate guide to PyTorch Optimizers](https://analyticsindiamag.com/ultimate-guide-to-pytorch-optimizers/)

9. [torch.optim](https://pytorch.org/docs/stable/optim.html)

10. [What is a PyTorch optimizer?](https://www.educative.io/answers/what-is-a-pytorch-optimizer)