


# 基于MobileNetV2的狗和牛角包二分类任务
2022年8月11日**更新**

在此教程中，我们将通过[drizzlezyk/MobileNetV2](https://xihe.mindspore.cn/projects/drizzlezyk/MobileNetV2)项目，快速体验图像二分类（狗和牛角包）任务的在线训练、推理。


# 目录  
[基本介绍](#基本介绍)  
- [任务简介](#任务简介)
- [项目地址](#项目地址)
- [项目结构](#项目结构)

[效果展示](#效果展示)
- [训练](#训练)
- [推理](#推理)

[快速开始](#快速开始)
- [Fork样例仓](#Fork样例仓)
- [在线推理](#在线推理)

[问题反馈](#问题反馈)



***
<a name="基本介绍"></a>
## 基本介绍

<a name="任务简介"></a>
### 任务简介

基于公开的模型仓库 drizzlezyk/MobileNetV2 进行模型训练，并使用仓库下的模型文件实现在线图像二分类推理。

#### MobileNetV2模型简介
MobileNetV1是为移动和嵌入式设备提出的轻量级模型。MobileNets使用深度可分离卷积来构建轻量级深度神经网络，进而实现模型在算力有限的情况下的应用。
MobileNet是基于深度可分离卷积的。深度可分离卷积把标准卷积分解成深度卷积(depthwise convolution)和逐点卷积(pointwise convolution)，进而大幅度降低参数量和计算量。深度可分离卷积示意图如下：
<img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/mobilenetv2/mobilenet_deep_conv.PNG" width="70%">

深度可分离卷积将标准的卷积操作(a)拆分为：(b)深度卷积和(c)逐点卷积，从而减少计算量。

MobileNetV2是在MobileNetV1的基础上提出一种新型层结构： 具有线性瓶颈的倒残差结构(the inverted residual with linear bottleneck)。模型结构如下：

<img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/mobilenetv2/mobilenetv2_model.PNG" width="80%">

MobileNetv2在极少的参数量下有着不错的性能。

#### 相关论文
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html)

    


其模型结构包括两个生成器G_X, G_Y，和两个判别器D_X, D_Y，结构图如下：

<img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/cycleGAN/cycleGAN-model.PNG" width="80%">

#### 数据集简介
使用的数据集包括两个类别的图片：**狗和牛角包**，数据集结构如下：
```
 ├── train    # 训练集
 │  ├── croissants   # 牛角包图片
 │  ├── dog   # 狗图片
 │── val   # 验证集
 │  ├── croissants   # 牛角包图片
 │  ├── dog   # 狗图片
```

<a name="项目地址"></a>
### 项目地址
- 模型仓库：[drizzlezyk/MobileNetV2](https://xihe.mindspore.cn/projects/drizzlezyk/MobileNetV2)
- 数据集地址：[drizzlezyk/MobileNetV2_image](https://xihe.mindspore.cn/datasets/drizzlezyk/MobileNetV2_image)
- 模型地址：[drizzlezyk/MobileNetV2_model](https://xihe.mindspore.cn/models/drizzlezyk/MobileNetV2_image)


<a name="项目结构"></a>
### 项目结构

项目的目录分为两个部分：推理（inference）和训练（train），推理可视化相关的代码放在inference文件夹下，训练相关的代码放在train文件夹下。

```python
 ├── inference    # 推理可视化相关代码目录
 │  ├── app.py    # 推理核心启动文件
 │  ├── pip-requirements.txt    # 推理可视化相关依赖文件
 │  ├── config.json
 └── train    # 在线训练相关代码目录
   ├── config.json  # 训练配置文件，用于指定代码路径、超参数、数据路径等
   └── train_dir         # 训练代码所在的目录
     ├── pip-requirements.txt  # 训练代码所需要的package依赖声明文件
     └── train.py       # 神经网络训练代码
```


***
<a name="效果展示"></a>
## 效果展示

<a name="训练"></a>
### 训练

 <img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/resnet50/trainlog.PNG" width="80%">

<a name="推理"></a>
### 推理
<img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/mobilenetv2/mobilenetv2_inference.PNG" width="80%">

***
<a name="快速开始"></a>
## 快速开始

<a name="Fork样例仓"></a>
### Fork样例仓

1. 在项目搜索框中输入MobileNetV2，找到样例仓 **drizzlezyk/MobileNetV2**

2. 点击Fork

<a name="在线训练"></a>
### 在线训练

创建训练后，就可以通过普通日志和可视化日志观察训练动态。

1. 选择“**训练**”页签，点击“**创建训练实例**”按钮，配置训练参数有两种方式：（1）通过json格式的配置文件创建；（2）在线填写表单。

2. 两种方式创建步骤如下：

   - 选择“**选择配置文件**”页签:在“**输入框**”输入训练配置模板的路径 **train/config.json**，点击确认按钮，train/config.json将会加载到下方表单，json文件中各个字段的注释如下，更多的详细介绍参考创建训练实例：

     ```json
     {	
     "SDK": "ModelArts",               	//训练平台，目前支持ModelArts
     "code_dir": "train/train_dir/",	  	//模型训练相关代码以及配置需求文件所在目录	
     "boot_file": "train.py",			//启动文件
     "outputs": [{						//训练过程需要输出的一些文件的路径设置
         "output_dir": "train/output/",	
         "name": "output_url"
     }],
     "hypeparameters": [{
        "name": "epochs",
		"value": "100"}],
     "frameworks": {						//模型所需的深度学习框架，目前支持MindSpore
         "framework_type": "MPI",
         "framework_version": "mindspore_1.3.0-cuda_10.1-py_3.7-ubuntu_1804-x86_64",
     },
     "train_instance_type": "modelarts.p3.large.public",	//计算资源
     "log_url": "train/log/",			//日志文件的保存路径
     "env_variables": {					//环境变量
     },
     "job_description": "训练MobileNetV2，epoch=100",    //对当前训练实例的描述
     "inputs": [
		{"input_url": "datasets/drizzlezyk/MobileNetV2_image/DogCroissants/",
		"name": "data_url"},
		{"input_url": "models/drizzlezyk/MobileNetV2/mobilenet_v2_1.0_224.ckpt", //预训练模型的存放路径
		"name": "pretrain_url"}],    
	 "job_name": "test-8-11"         //训练名称
     }
     ```

     为了在更快的时间训练出更好的模型，建议您选择使用[models/drizzlezyk/MobileNetV2](https://xihe.mindspore.cn/models/drizzlezyk/MobileNetV2) 下预训练模型mobilenet_v2_1.0_224.ckpt进行微调。需要在input_url下加上如下超参数：

        ```json
        {"input_url": "models/drizzlezyk/MobileNetV2/mobilenet_v2_1.0_224.ckpt",
        "name": "pretrain_url"}     //预训练模型的存放路径
        ```
   - 选择填写表单方式创建训练:将json文件中对应的值填到表单即可。
    
3. 点击创建训练，注意一个仓库同时只能有一个运行中的训练实例，且训练实例最多只能5个。

4. 查看训练列表：将鼠标放置于“**训练**”栏上，点击训练下拉框中的“**训练列表**”即可。
  
   <img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/resnet50/resnet-train-list.PNG" width="80%">

 

5. 查看训练日志：点击训练名称，即可进入该训练的详情页面：
   <img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/resnet50/trainlog.PNG" width="80%">


<a name="在线推理"></a>
### 在线推理

本项目的推理模块是将训练好的模型应用到实时的图像风格迁移任务中，可将一些生活中自然风光的照片和艺术家的画作进行相互的风格迁移，具体如下：


1. 选择“**推理**”页签，点击“**启动**”按钮。

2. 等待2分钟左右，出现推理可视化界面，提交一张狗或者牛角包的图片，点击submit进行预测：

     <img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/mobilenetv2/mobilenetv2_inference.PNG" width="70%">

***
<a name="问题反馈"></a>
# 问题反馈

本教程会持续更新，您如果按照教程在操作过程中出现任何问题，请您随时在我们的[官网仓](https://gitee.com/mindspore/mindspore)提ISSUE，我们会及时回复您。如果您有任何建议，也可以添加官方助手小猫子（微信号：mindspore0328），我们非常欢迎您的宝贵建议，如被采纳，会收到MindSpore官方精美礼品哦！
