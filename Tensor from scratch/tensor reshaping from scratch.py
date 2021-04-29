import torch

# ================================================================== #
# Tensor reshaping                                                   #
# ================================================================== #

x = torch.arange(9)

x_3x3 = x.view(3,3)  ## memory should be contiguous
print(x_3x3)
x_3x3 = x.reshape(3,3) ## memory could be not contiguous( always works, but have some performance loss )
print(x_3x3.shape)

y = x_3x3.t()
print(y.contiguous().view(9)) # 왼쪽코드 transpose해서 contiguous하지 않아서 에러가 발생한다

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=1).shape)

# unrolling

z = x1.view(-1)
z = x1.flatten()

batch = 64 

x = torch.rand((batch, 2, 5, 5))
x = x.view(batch, -1)
print(x.shape)

z = x.permute(0,2,1,3) # 참고 : .t 트랜스포는 permute의 special case이다.
print(z.shape) 

x = torch.arange(10).unsqueeze(0).unsqueeze(1)
x = x.squeeze(1)
print(x.shape)
