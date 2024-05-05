# torch._tensor.py
class Tensor(torch._C._TensorBase):
# torch._C.__init__pyi
class _TensorBase(metaclass=_TensorMeta):
	...
	def __mul__(self, other: Any) -> Tensor: ...

