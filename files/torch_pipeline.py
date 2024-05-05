# torch._tensor.py
class Tensor(torch._C._TensorBase):
# torch._C.__init__pyi
class _TensorBase(metaclass=_TensorMeta):
	...
	def __mul__(self, other: Any) -> Tensor: ...

import torch
from torch import optim
'''前向计算过程'''
x = torch.tensor([0.5, 0.75], requires_grad=True)
v = x[0] * x[1]
print(v)
# tensor(0.3750, grad_fn=<MulBackward0>)
'''反向传播计算梯度'''
v.backward()
'''使用优化器更新参数'''
optimizer = optim.SGD([x[0], x[1]], lr=0.1)
optimizer.zero_grad()
optimizer.step()