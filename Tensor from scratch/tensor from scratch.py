import torch

# ================================================================== #
# initializing Tensor                                                #
# ================================================================== #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
v = torch.tensor([1,2,3],dtype = torch.float32, device=device,requires_grad=True)


# other common initialization
x = torch.empty(size = (3, 3))
x = torch.zeros((3, 3))
x = torch.rand((3, 3))
x = torch.ones((3, 3))
x = torch.eye(5, 5)
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
x = torch.empty(size=(1, 5)).uniform_(0, 1)
x = torch.diag(torch.ones(3))

# how to initialize and convert tensors to other types (int, float, double)
v = torch.arange(4)
print(v.bool()) # boolean
print(v.short()) # int16
print(v.long()) # int64
print(v.half()) # float16
print(v.float()) # float32
print(v.double()) # float64

# array to tensor conversion
import numpy as np
a = np.zeros((5, 5))
t = torch.from_numpy(a)
a_b = t.numpy()

# ================================================================== #
# Tensor math                                                        #
# ================================================================== #

t = torch.tensor([1, 2, 3])
t2 = torch.tensor([9, 8, 7])

# add, subtrack, divide, mul (elementwise)
# skip

# inplace operation

t.add_(t2)
t += t2

# Exponentiation
z = t.pow(2)
z = t ** 2 
print(z)

# matrix multiplicate

t1 = torch.rand((2,5))
t2 = torch.rand((5,3))
t3 = torch.mm(t1,t2)
t3 = t1.mm(t2)

# matrix exponentiation
m_e = torch.rand(5, 5)
print(m_e.matrix_power(3))

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1,tensor2) # (batch, n, p)

# Example of boradcasting

x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2
z = x1 ** x2 # x2 will be copied all over

# Other usedful torch math

sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0) 
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0) # torch.max 와 비슷하지만 index만 반환한다
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim =0)
z = torch.eq(t1, t2) ## 같으면 True 다르면 False
sorted_t2, indices = torch.sort(t2, dim=0, descending=False) ## ascending 으로할꺼임 -> increasing order
z = torch.clmap(x, min=0, max=10) # 한계를 정해서 넘거나 작으면 그값으로 정한다

x = torch.tensor([1,0,1,1,1])
z = torch.any(x) # True 하나라도 True니까,,
z = torch.all(x) # False 전부다 True는 아니니까,,



