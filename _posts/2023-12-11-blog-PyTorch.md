---
title: 'PyTorch 随笔'
data: 23-12-11
permalink: '/post/2023/12/blog-pytorch'
tags:
  - 深度学习基础知识
---

```torch.backends.cudnn.deterministic```: 固定 cuda 的随机种子，使得每次返回的卷积算法都是确定的，即默认算法

```python
torch.manual_seed(seed)  # sets the seed for generating random numbers.
torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU.
torch.cuda.manual_seed_all(seed)  # Sets the seed for generating random numbers on all GPUs.
```

```torch.backends.cudnn.benchmark```: 会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速

DataLoader中的pin_memory就是锁页内存(锁页内存存放的内容在任何情况下都不会与主机的虚拟内存交换)，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转移到GPU的显存就会更快一些

```dist.init_process_group``` 中的参数

- ```backend```：后端数据传输的模式，一般情况下 multi-CPU 使用 ```gloo```，multi-GPU 使用 ```nccl```；而 ```mpi```需要额外的下载配置

- [```init_method```：定义了GPU之间的通信模式](https://pytorch.org/docs/stable/distributed.html#tcp-initialization)
  
  - ```dist.init_process_group```的 ```init_method```默认参数为```env://```，此时需要设置
  
  ```python
  os.environ["MASTER_ADDR"] = "master的地址, 对于单机多卡, 一般是localhost; 而对于多机多卡, 一般是 master:0 的机器的地址"
  os.environ["MASTER_PORT"] = "端口号"
  # 对于单机多卡, 可以不进行 addr 和 port 的设置, 则此时默认为 localhost 和 29500
  # 当使用 torch.multiprocessing 启动时, 需要提供 world_size, rank
  import torch.multiprocessing as mp
  def example(rank, world_size):
      # create default process group
      dist.init_process_group("gloo", rank=rank, world_size=world_size)
      # create local model
      model = nn.Linear(10, 10).to(rank)
      # construct DDP model
      ddp_model = DDP(model, device_ids=[rank])
  world_size = 4
  mp.spawn(example,
      args=(world_size,),
      nprocs=world_size,
      join=True)
  # 当使用 torch.distributed.launch 启动时, 无需提供 world_size, rank, 但需要在命令行提供 --nproc_per_node, 同时需要指定额外参数 --local-rank(不需要自己传参)
  def example():
      parser = argparse.ArgumentParser()
      parser.add_argument('--local_rank', default=0, type=int)
      args = parser.parse_args()
      dist.init_process_group(backend="gloo", init_method='env://')
      rank = args.local_rank
  python -m torch.distributed.launch --nproc_per_node=$你需要用的GPU数$ yourfile.py
  
  # 无论是从 torch.multiprocessing 启动并传入参数 rank, 还是从 torch.distributed.launch 启动 传入 local_rank, 在模型和数据 forward 时都需要将其转移到上面：
  # 下面将 rank 和 local_rank 统称为 rank
  model = model.to(rank)
  ddp_model = DDP(model, device_ids=[rank])
  data, label = data.to(rank), label.to(rank)
  # 如果直接 .cuda(), 则默认会将所有数据放置在 cuda:0, 但是由于模型经过 DDP 的包装并加上device_ids=[rank]后, 复制到了各个GPU上, 这时进行 model(data.cuda()) 会报错数据与模型参数不在同一个 device 上; 如果要使用 .cuda(), 则需要在开始时设置：
  torch.cuda.set_device(rank)
  # 或者将 device_ids=[rank] 除去, 使得程序自动判断
  # 注意：torch.distributed.launch 启动 和 dist.init_process_group 中的 world_size 设置不要和 --nproc_per_node 一起/或者不能不一致, 否则会卡住; 同时, 不能只设置 world_size 而不设置 rank 和 不加 --nproc_per_node, 也会导致卡住
  ```
  - ```tcp```：
  
  ```python
  dist.init_process_group(backend="gloo", init_method='tcp://addr:port', rank=rank, world_size=world_size)
  # tcp 必须指定 rank 和 world_size 以及 addr 和 port, 其可以通过 os.environ 获得：
  rank = int(os.environ["LOCAL_RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  addr = os.environ["MASTER_ADDR"]
  port = os.environ["MASTER_PORT"]
  # 也可以自己设置 world_size, addr 和 port; 而 rank 可以通过 args 添加 --local_rank 使得程序自动输入, 并通过 args.local_rank 获得
  # 注意：torch.distributed.launch 启动 和 dist.init_process_group 中的 world_size 设置不要和 --nproc_per_node 一起/或者不能不一致, 否则会卡住; 所以可以自己设置 world_size 而不传入参数 --nproc_per_node
  ```

```torchrun``` 启动

```python
torchrun --nnodes=$机器数$ --nproc_per_node=$每个机器的GPU数$ --rdzv_id=$$ --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:MASTER_PORT$ yourfile.py
# 此时需要注意：arg 参数不需要再设置 --local_rank
# 如果初始化方法为 tcp, 需要地址+端口+rank+world_size, 则可以从 os.environ 获取
rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
addr = os.environ["MASTER_ADDR"]
port = os.environ["MASTER_PORT"]
```
