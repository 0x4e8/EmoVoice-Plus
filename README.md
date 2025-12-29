## 准备
### 环境、权重、数据
首先按照 [EmoVoice](https://github.com/yanghaha0908/EmoVoice) 配置环境、下载预训练模型权重以及情感微调数据集。

### 下载用于计算情感激活方向的数据集

### 生成情感激活方向
```
bash examples/tts/scripts/gene_activate_pt.sh
```
### 离线计算语音情感特征
```
bash examples/tts/scripts/gene_style_emb.sh
```
## 训练
```
bash examples/tts/scripts/ft_EmoVoice-PP-align.sh
```
训练脚本中，```train_config.align.enable=true```表示使用交叉熵损失+对比损失进行训练，```train_config.align.enable=false```表示使用仅交叉熵损失进行训练。

可以下载已经训练好的模型权重：
## 推理
```
bash examples/tts/scripts/inference_EmoVoice-PP.sh
```
推理脚本中，```decode_config.emosteer.enable=true```表示使用情感激活注入，```decode_config.emosteer.enable=false```表示不使用情感激活注入。

