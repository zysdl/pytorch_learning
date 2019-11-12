# import torch
# from matplotlib import pyplot as plt
import numpy as np
import torch
from numpy.core.multiarray import ndarray
#数据的输入
#torch.rand是随机0到1均匀分布，randn是随机正态分布
# torch.tensor(2)
# print(torch.tensor([2.3,4,56]))#小数点后面多了一个三，就默认变为四位小数，最多只有四位
#
# data=numpy.ones(2)
# print(data,torch.FloatTensor(2))
#
#
# print(torch.from_numpy(data),"\n")
#
# a=torch.FloatTensor(2,3,)
# print(a,"\n")
# print(a.shape,a.size(0))#shape和size是一样的，都表示在第几维上的东西有多少个
# print(a.shape,a.size(1))
#
# a=torch.rand(1,2,3)
# print(a,   a.shape,"\n","\n")
# print(a[0])         #tensor的引用
# print(list(a.shape))#强制从shape类型转换成python里面的list类型
#
# print(a.numel())    #numel输出a所占内存大小

#选中，然后ctrl加/可以多行注释

#数据的引入：从numpy到tensor：方便用GPU加速运算
a=np.array([2,3.3])

print(torch.from_numpy(a))#导入以后会自动变成double类型

a=np.ones([2,3])
print(torch.from_numpy(a))

print(torch.tensor([2.,3.2]))#小写的是接受现有数据，大写是接受作为shape，即维度，并且没有初始化，随机的
print(torch.FloatTensor(2,2))  #Tensor会默认转化为IntTensor或者其他
print(torch.FloatTensor([2,2]))#这样就可以接受list而不是作为shape
torch.set_default_tensor_type(torch.FloatTensor)#设置默认Tensor数据类型
print(torch.tensor([1,2]).type())


# 未初始化方法
torch.empty(1)#接受的是shape
# torch.FloatTensor()或者IntTensor
#未初始化的后面需要赋值
# 随机会导致数据很大或者很小很不规则

#初始化方法
print(torch.rand(3,3))   #均匀采样，但不是顺序分布

a=torch.rand(3,3)
print(torch.rand_like(a))#接收一个tensor，随机初始化在0到1

print(torch.randint(1,12,[5,2]))#整数随机，包括1，不包括12

torch.randn(3,3)#0到1的正态分布
#设置均值和方差的方法
# print(torch.normal(mean=torch.full([20],0),std=torch.arange(0,2,0.1)))#mean是均值，10个初值为0的tensor，std为方差，从0到1方差越来越小
#
# torch.full([],7)#生产一个标量，7
# torch.full([1],7)#一个向量，7
#
# print(torch.arange(0,100,2))#从0到100，增大2。但是不包含100
# print(torch.linspace(0,10,steps=7))#把0到10均等切割为7份
# print(torch.logspace(1,10,steps=10))#对数划分法

print(torch.ones(3,3))#填充1，.zero就是填充0

print(torch.eye(3,4))#在对角中填充1，若对角不在中间，就剩下的填0
torch.eye(3)#会自动生产3*3的对称矩阵

print(torch.randperm(10))#生产连续数，随机打散

a=torch.rand(2,3)
b=torch.rand(2,2)
idx=torch.randperm(2)
print(idx,a[idx],a,b)#因为idx随机，所以，从a中取出来的顺序也不同。a[0,1],a[1,0]，表示分别取a[0],a[1]and,a[1],a[0]






