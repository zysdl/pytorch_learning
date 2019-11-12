import torch
a=torch.rand(4,3,28,28)
#规则采样


# print(a[0].shape)
print(a[:2].shape)#从0开始，到1，不包括2
print(a[:2,:2,:7,:4])#[:,:,:]中括号里面的每一个逗号都是隔了一个dim，一个维度
#如果是赋初值，中括号是dim为1，的list
print(a[:2,:2,27:,27:]) #start:end：jiange，不包含末尾,最后一个是隔多少个采样一次的一次，默认为1。

#自定义采样
print(a.index_select(0,torch.tensor([0,2])).shape)
print(a.index_select(1,torch.tensor([1,2])).shape)
print(a.index_select(2,torch.arange(8)).shape)  #index_select(维度，维度里面的具体哪些个tensor)

#全部采样
print(a[0,...,:14].shape)  #三个点仅仅是为了方便

#掩码采样   （弊端：默认把数据打平）
x=torch.randn(3,4)
mask=x.ge(0.5)
print(mask=a.ge(0.5))    #ge是great的意思吗
print(torch.masked_select(x,mask))

#打平后采样
src=torch.tensor([[4,3,5],[6,8,9]])
torch.take(src,torch.tensor([0,2,6]))


