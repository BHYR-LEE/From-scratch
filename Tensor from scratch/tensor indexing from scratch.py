import torch

# ================================================================== #
# Tensor indexing                                                    #
# ================================================================== #

batch_size = 10
features = 20

x = torch.rand((batch_size, features))

print(x[0]) # x[0,:]

print(x[:,0]) # first feature of all examples
print(x[2,0:10]) # 3rd example's 10 features

x[0,0] = 100 # assing 도 가능하다

# Fancy indexing

x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print((x[rows,cols]))

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8) ])
print(x[x.remainder(2) == 0])

## Useful operations
print(torch.where(x > 5, x, x*2))
print(torch.tensor([0,0,1,2,3,4]).unique())
print(x.ndimension())
print(x.numel())
