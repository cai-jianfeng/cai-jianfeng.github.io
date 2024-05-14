# torch._tensor.py
class Tensor(torch._C._TensorBase):
# torch._C.__init__pyi
class _TensorBase(metaclass=_TensorMeta):
	...
	def __mul__(self, other: Any) -> Tensor: ...

import torch
from torch import optim
'''前向计算过程'''
x1 = torch.tensor([0.5], requires_grad=True)
x2 = torch.tensor([0.75], requires_grad=True)
v = x1 * x2
print(v)
# tensor(0.3750, grad_fn=<MulBackward0>)
'''反向传播计算梯度'''
v.backward()
'''使用优化器更新参数'''
optimizer = optim.SGD(params=[x1, x2], lr=0.1)
optimizer.zero_grad()
optimizer.step()

class SGD:
	def __init__(self, params, lr) -> None:
		self.params = params
		self.lr = lr

	# 在 torch 代码里，是通过 optimizer.step() 来实现参数更新
	def step(self):
		with torch.no_grad():
			for param in self.params:
				if param.required_grad:
					grad = param.grad
					# 这里需要原地更新，不能使用 param = param - grad * self.lr
					# 这里科普一个小知识点：`-=`` or `+=`` or `*=`` or `/=` 操作对于类对象是原地操作
					# 类对象是指非基础对象，比如 列表、元组、字典等
					param -= grad * self.lr

class SGD(Optimizer):
	def __init__(self, params, lr, ...):
		...
		defaults = dict(lr=lr, ...)
		super().__init__(params, defaults)
	
	@_use_grad_for_differentiable
	def step(self, closure=None):
		for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            sgd(params_with_grad,
                d_p_list,
                ...
                lr=group['lr'],
                ...)
	
	def _init_group(self, group, params_with_grad, d_p_list, ...):
		...
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
				...
	
def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        ...
        lr: float,
        ...):
	...
    if foreach and not torch.jit.is_scripting():
        ...
    else:
        func = _single_tensor_sgd

    func(params,
         d_p_list,
         ...
         lr=lr,
         ...)

def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       ...
                       lr: float,
                       ...):

    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]
		...
        param.add_(d_p, alpha=-lr)
		

class Optimizer:
	def __init__(self, params: params_t, defaults: Dict[str, Any]) -> None:
		...
		self.state: DefaultDict[torch.Tensor, Any] = defaultdict(dict)
        self.param_groups: List[Dict[str, Any]] = []
		...
		param_groups = list(params)
		...
		for param_group in param_groups:
            self.add_param_group(cast(dict, param_group))

	@torch._disable_dynamo
    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        ...
        params = param_group['params']
		...
        self.param_groups.append(param_group)
	