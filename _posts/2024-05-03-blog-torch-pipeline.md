1. optimizer 中的 self.param_groups 和 self.states 的 keys 都是与 model.parameters() 共享内存空间，即它们都指向同一个内存区域

2. dict 的 keys(), values() 和 items() 的返回值与 dict 共享内存空间，对其值进行“原地”操作会同步修改 dict 内的值











# Reference

1. [Understanding PyTorch with an example: a step-by-step tutorial](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)

2. [PyTorch 的 SGD 源码](https://github.com/pytorch/pytorch/blob/cd9b27231b51633e76e28b6a34002ab83b0660fc/torch/optim/sgd.py#L63)