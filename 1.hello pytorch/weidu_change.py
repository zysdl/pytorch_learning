import torch
# #view and reshape   维度变换
# #view and reshape 是一样的。变换以后保证numel最终是一样的就行
# a=torch.rand(4,1,28,28)
# print(a.view(4,28*28))
# print(a.view(4 * 28, 1,28))
# #当数据被降维，再升维，必须按照原来的b,c,h,w的顺序来，否则就不再是原来的顺序了。
#
# #squeeze and unsqueeze 压缩和展开
# b=torch.rand(4,1,28,28)
# a.unsqueeze(0)#在0维插入，可以在-5到4之间取值
# #不会增加数据，只是增加组别

c=torch.rand(12)
c=c.unsqueeze(0).unsqueeze(2).unsqueeze(3)
print(c.shape)
#squeeze  压缩，减小维度
c=c.squeeze()  #也可以接受负的索引
print(c.shape)

#维度拓展  expand：有需要才复制 and repeat：先复制
#必须维度一样，而且，只有原来是1的，才可以拓展
# c.expand([4,32,4,4])#这样就拓展好了
# c.repeat(4,1,4,4)#在各个维度上重复的次数，而不是最终变成的维度。   不推荐使用
#转置方法 a.t() 只适用于二维

#自由维度转置  会把在内存中的顺序打乱，所以需要contiguous复制一下
d=torch.rand(1,3,3)
print(d,d.shape)
a1=d.transpose(0,2)
print(a1)
a1.contiguous()#确实是要有这个叫做什么contiguous的函数才能用呢。
print(a1,a1.shape)
# d.view(2,6).view(3,2,2)
a1.transpose(0,2).contiguous()
print(torch.all(torch.eq(d,a1)),a1.shape)

a1.permute(2,1,0)#可以按照任意顺序，多次调用transpose直到达到目的为止
print(a1)









