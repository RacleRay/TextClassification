HAN参照原论文[Hierarchical Attention Networks for Document Classification](https://www.aclweb.org/anthology/N16-1174)

实现，模型结构如下：

![1566401953564](assets\1566401953564.png)

对这一结构的改进：Hierarchical Attentional Hybrid Neural Networks for Document Classification，2019，arXiv:1901.06610v2

![1566402216531](assets\1566402216531.png)

主要不同是使用了Temporal Convolutional层，来进一步提取特征词依赖特征。

![1566402299659](assets\1566402299659.png)

主要缺点就是inference时间进一步增加，准确率在二分类问题上的提升相对较高一些。
