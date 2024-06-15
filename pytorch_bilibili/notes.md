### code

https://blog.csdn.net/bit452/category_10569531.html

### video

https://www.bilibili.com/video/BV1Y7411d7Ys/?p=3&spm_id_from=pageDriver&vd_source=5b38429dc2a3c27f4efb082ccdfe871a

===============

1、softmax 的输入不需要再做非线性变换，也就是说 softmax 之前不再需要激活函数(relu)。softmax 两个作用，如果在进行 softmax 前的 input 有负数，通过指数变换，得到正数。所有类的概率求和为 1。

2、y 的标签编码方式是 one-hot。我对 one-hot 的理解是只有一位是 1，其他位为 0。(但是标签的 one-hot 编码是算法完成的，算法的输入仍为原始标签)

3、多分类问题，标签 y 的类型是 LongTensor。比如说 0-9 分类问题，如果 y = torch.LongTensor([3])，对应的 one-hot 是[0,0,0,1,0,0,0,0,0,0].(这里要注意，如果使用了 one-hot，标签 y 的类型是 LongTensor，糖尿病数据集中的 target 的类型是 FloatTensor)

4、CrossEntropyLoss <==> LogSoftmax + NLLLoss。也就是说使用 CrossEntropyLoss 最后一层(线性层)是不需要做其他变化的；使用 NLLLoss 之前，需要对最后一层(线性层)先进行 SoftMax 处理，再进行 log 操作。
