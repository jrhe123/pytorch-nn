### video: (bert / transformer / self-attention)

- Query / Key / Value
- scaled dot-product attention 内积
- softmax

- https://www.bilibili.com/video/BV1tm4y187b5?p=92&vd_source=5b38429dc2a3c27f4efb082ccdfe871a

1. 卷积：提取特征
2. 池化：压缩特征

- max pooling: 取最大值 (取最重要的特征) => down sampling
- avg pooling: 取平均值

3. TL: 迁移学习 (transfer learning)

- 冻住一部分 layers (比如所有卷积层), 只训练最后一层 （全连接层）softmax
  - 训练数据量小，可以只训练最后一层
  - 训练数据量大的时候，可以训练所有层
- 通常！先学一部分，再学全部的

4. RNN: recurrent neural network / LSTM: long short-term memory

- Word2Vec: 50 - 300 dimension
- CBOW / Skipgram
- 负采样：手动生成，结果为 0
