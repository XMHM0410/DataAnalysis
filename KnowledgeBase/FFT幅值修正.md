# FFT幅值修正

<img src="C:\Code\DataAnalysis\KnowledgeBase\fig1.jpg" alt="img" style="zoom:75%;" />

<img src="C:\Code\DataAnalysis\KnowledgeBase\fig2.jpg" alt="img" style="zoom:75%;" />

<img src="C:\Code\DataAnalysis\KnowledgeBase\fig3.jpg" alt="img" style="zoom:75%;" />

最后将DFT进行改进成FFT，即FFT是DFT的[快速算法](https://www.zhihu.com/search?q=快速算法&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2962903544})，本质上仍然是DFT。

MATLAB中提供“fft”函数，可以直接对数据进行快速傅里叶变换。但是由于FFT的本质仍然是DFT，则得到的频谱是用[功率谱密度](https://www.zhihu.com/search?q=功率谱密度&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2962903544})（PSD）定义的，也就是它的幅值表示的是单位带宽的幅值。

N：[样本点](https://www.zhihu.com/search?q=样本点&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2962903544})

FsF_{s}F_{s} ：[采样频率](https://www.zhihu.com/search?q=采样频率&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2962903544})



FFT变换之后的横轴为频率轴，[频谱图横坐标](https://www.zhihu.com/search?q=频谱图横坐标&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2962903544})显示的最大频率点为 Fs2\frac{F_{s}}{2}\frac{F_{s}}{2} （[奈奎斯特采样定理](https://www.zhihu.com/search?q=奈奎斯特采样定理&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2962903544})），频率的坐标间隔（频率分辨率）为 FsN\frac{F_{s}}{N}\frac{F_{s}}{N} ，也就是说最小能分辨的两个频率之间的差值要大于 FsN\frac{F_{s}}{N}\frac{F_{s}}{N} 。

  横轴的频率点为：（0：1 ⋅FsN\cdot\frac{F_{s}}{N}\cdot\frac{F_{s}}{N} ： N2⋅FsN\frac{N}{2}\cdot\frac{F_{s}}{N}\frac{N}{2}\cdot\frac{F_{s}}{N} ）。（经过FFT之后，频谱是关于中间位置对称的，只需要观察0~ N2\frac{N}{2}\frac{N}{2} 即可）

对于实数信号而言，N（先假设N为偶数，其实最好的情况是为2的n次幂个）个离散点的DFT将产生 N2\frac{N}{2}\frac{N}{2} +1个频率点（因为无论怎样都会有0频率），频率的序号从0~ N2\frac{N}{2}\frac{N}{2} 。故而N个[实数点](https://www.zhihu.com/search?q=实数点&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2962903544})经过DFT之后的频谱带宽为 N2\frac{N}{2}\frac{N}{2} ，每一个频率点所占的带宽为 2N\frac{2}{N}\frac{2}{N} ，所以经过FFT之后的频率对应的幅值并不是真实的幅值，想要将FFT之后的频谱图跟实际的幅值对应，就要进行相应的转换。

那么，重点来了，怎么转化呢？

上边我们说的就是每一个频率点所占的带宽是 2N\frac{2}{N}\frac{2}{N} ，将带宽跟FFT之后得到的幅值相乘即可对应到原本的幅值。但是，需要特别注意的一点：

频率序号为0和 N2\frac{N}{2}\frac{N}{2} 的两个点的带宽只占中间频率点的一半，也就是占 1N\frac{1}{N}\frac{1}{N} 的带宽，因此只需要将幅值乘以 1N\frac{1}{N}\frac{1}{N} 即可。

到这，也许有细心的童鞋发现了， N2\frac{N}{2}\frac{N}{2} 这个数据对应的N为偶数是最好的，那如果恰好N为奇数又该怎么解决呢？

事实上，当我们的样本点N为奇数时，只有0频率占 1N\frac{1}{N}\frac{1}{N} 带宽，其余的 N+12\frac{N+1}{2}\frac{N+1}{2} 个频率点仍然是对应幅值乘以 2N\frac{2}{N}\frac{2}{N} 得到真实的幅值。

以上都是基于实数信号，对于[复数信号](https://www.zhihu.com/search?q=复数信号&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2962903544})而言（生活中大部分的[模拟信号](https://www.zhihu.com/search?q=模拟信号&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2962903544})转化为数字信号都是实数信号，很少遇到复数信号），处理方法则相对简单多了。对于复数信号，N个点FFT之后会产生N个频率点，频谱的带宽为N，每个点所占的带宽为 1N\frac{1}{N}\frac{1}{N} ，将每个幅值都乘以 1N\frac{1}{N}\frac{1}{N} 即可得到真实的频率幅值。
链接：https://www.zhihu.com/question/30501507/answer/2962903544。