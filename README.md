# wenet_mnn
只是在开源的wenet基础上，使用阿里的MNN推理框架进行部署的示例，一共两个部分，分别是Python对ONNX和MNN的调用，以及使用C++对MNN的调用。

测试使用的onnx/mnn模型下载地址：
链接：https://pan.baidu.com/s/1c3UbfdvUrtNUotC_O_3qFA   
密码：02h3

详细的介绍，可以参考知乎文章：https://zhuanlan.zhihu.com/p/559531090?

C++的解码效果，见下图：

![0975cac279760e19eabffed371a6900](https://user-images.githubusercontent.com/46361879/187656143-157fe8cd-6771-4958-803d-0201608bcf2c.png)

Python的解码效果如下：

![image](https://user-images.githubusercontent.com/46361879/187656325-3c9e3902-cd90-4cfd-9497-a4c5172c058e.png)
