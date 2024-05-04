from torch.autograd.function import Function
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
#Use it by calling the apply method:
# 使用 Function 函数时，应调用 .apply() 函数代替 .forward() 函数，无法直接调用 .forward() 函数
output = Exp.apply(input)

from torch.autograd import Function
class AccumulateGrad(Function):
    @staticmethod
    def forward(ctx, i):
        # 前向过程，即复制输入数据
        result = i
        # 保存输入数据，以便在后向时可以找到它并将梯度写入其 .grad 属性
        ctx.variable = result
        return result
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.variable
        # 将梯度写入输入数据的 .grad 属性
        result.grad = grad_output
        return grad_output
    
op_map ={'identity':AccumulateGrad(), ...}

def computation_graph_build(inputs, operations):
    for input in inputs:
    # 首先对所有输入判断是否需要梯度计算:
    # 如果需要，则将其 grad_fn 注册为 AccumulateGrad 来将最终的梯度存入 .grad 属性
        if input.requires_grad:
            input.grad_fn = AccumulateGrad()
            input.grad_fn.variable = input
    for operation in operations:
        for in_datas, op in operation:
            # 对于所有操作，in_datas 表示其输入数据，op 表示其操作:
            # 数据操作类型选择合适的 Function 子类
            op = op_map[op]
            # 将输入数据的 grad_fn 作为当前 Function 实例的前置节点，即构建计算图
            op.next_functions = [{in_data.grad_fn: num_unib} for in_data in in_datas]
            # 前向计算模型的输出
            out_datas = op.apply(in_datas)
            # 将当前操作注册进 grad_fn 属性
            out_datas.grad_fn = op
            # TOD0:这果省略了如何将 out_datas 作为下一个操作的输入
    # 最后一个输出即为根节点 loss
    loss = out_datas
    return loss