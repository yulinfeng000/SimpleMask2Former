# SimpleMask2Former

将Mask2Former网络结构与框架剥离,在内部数据集上达到和原论文实现相同精度

## Notice 说明

- 本代码只复现了ResNet50以及实例分割任务

## Reproduce Detail 复现细节与发现

1. 若不加载初始权重,训练过程中极易发生 loss 为 nan 的情况,故将原始论文权重作为初始权重

2. backbone 作为任意替换的模块,直接照搬了原论文的 ResNet50 网络，并做了相应适配

3. 复现过程中发现 optimizer 十分关键,其中针对 backbone 反向传播的梯度做0.1的缩放处理可以视为变相的冻结 backbone 权重,这会使得模型收敛更稳定


## Usage 使用方法

1. 下载 [Mask2Former-ResNet50](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_R50_bs16_50ep/model_final_94dc52.pkl) 权重到根目录

2. 修改train.py中COCO格式数据集的对应路径

3. python train.py 启动训练