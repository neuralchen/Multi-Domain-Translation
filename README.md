# Patch based HQ Face attributes editing algorithm

# Dependencies:
- PyTorch1.2 We recommend to use PyTorch1.2 or higher version, since lower version will meet the caculation problem of the gradient penalty in multi-GPU situation.
- numpy
- scipy
- tensorboardX
- tqdm
- pynvml  ` pip install pynvml `
- pytorch 1.1及以上版本

# modify:
这一版已经相较之前修改了许多，现在的代码需要指定生成器的脚本、判别器的脚本、分类器的脚本，这样做是因为此前大量更改这些脚本为了调试方便我们代码目前直接使用Python的反射机制根据脚本名载入相应的代码，这些脚本在运行时候会自动的保存到当前实验的日志的scripts文件夹中，此后再次断开后接着训练我们的工程会从scripts中读取这四个脚本，这样做是为了日后能恢复此实验。
现在的日志是自动保存在`./TrainingLogs/[当前实验version]/`目录下，其中：
`/checkpoint` 保存模型，需要保存的模型有判别器、生成器、G的学习率、D的学习率
`/sample` 采样的图片
`/scripts` 保存相关重要的脚本
`/summary` 日志
`config.json`是整个试验的设置保存，这个文件主要用来实现断开后继续训练各种配置的读取，还有测试模型时候各种配置的读取，现在因为设置的东西太过于多，所以我们将其保存起来时候去读取这样不会弄错。


# Usage:
## train
```python main.py --mode train --version [train_name] --TrainScriptName [your script name]```

### Attention
训练脚本存放在 `TraininScripts/` 文件夹中，自己的脚本命名方式 `trainer_[your name].py`


### `trainer_[your name].py` 代码结构为：
- `__init__`函数，主要完成parameter里的参数传值，调用`self.build_model()`完成构建模型、优化器、初始化summary
- `build_model`函数，主要完成模型的构建，优化器的构建，优化器现在改为`list`的形式，将需要与生成器一起优化的放在`gOptimizerList`中，将需要与判别器一起优化的放在`dOptimizerList`中
- `dResetGrad`函数，清零`dOptimizerList`中的梯度，这个每次调用`loss.backward()`之前必须调用，这是因为在pytorch中梯度是每次反传累加的，所以必须清零
- `gResetGrad`函数，清零`gOptimizerList`中的梯度，这个每次调用`loss.backward()`之前必须调用
- `dOptimizerStep`函数，更新D权值
- `gOptimizerStep`函数，更新G权值
- `train`函数，训练代码
- `DEBUG`标志位，在代码最顶上，这一位用来显示显存占用情况

### 需要解释模块：
#### `./utilities/CheckpointHelper.py`：
- `loadPretrainedModel`函数：用来载入checkpoint,
        参数列表：
        `chechpointStep`  --  checkpoint step for the pretrained model
        `modelSavePath`   --  the patch to save the trained model
        `gModel`          --  生成器backbone模型
        `dModel`          --  判别器backbone模型
        `**kwargs`        --  其他可能加入的模型字典例如：ToRGB，输入形势为`dict={"modelname1":model1,"modelname2":model2...}`
        样例：
        `params = {"GfRGB":self.GfRGB,"GtoRGB":self.GtoRGB,"DfRGB":self.DfRGB}`
        `loadPretrainedModel(self.chechpoint_step,self.model_save_path,self.GlobalG,self.GlobalD,**params)`
- `saveModel`函数：用来保存checkpoint，使用方法与`loadPretrainedModel`函数一致

#### `./utilities/ActivatorTable.py`：
- `getActivator`函数，通过激活函数名获得激活函数，包括激活函数：`relu,selu,leakyrelu,hardtanh`

# Parameters
|  Parameters   | Function  |
|  :----  | :----  |
| --version  | Experiment name |
| --TrainScriptName | 训练脚本的后缀，脚本命名方式`trainer_[your name].py`，此项输入为`[your name]`即可，训练脚本存放在`./TrainingScripts`文件夹中 |
| --TestScriptsName | 测试脚本的后缀，脚本命名方式`tester_[your name].py`，此项输入为`[your name]`即可，测试脚本存放在`./TesterScripts`文件夹中 |
| --mode  | Set the model stage, train, finetune, test |
| --imsize  | 输入图片的大小 |
| --imCropSize  | 输入图片crop大小 |
| --g_conv_dim  | 生成器的输入维度 |
| --GEncActName  | 生成器Encoder部分的激活函数名称可选"relu,selu,leakyrelu" |
| --GOutActName  | 生成器最后一个输出部分的激活函数名称可选"tanh,hardtanh" |
| --GDecActName  | 生成器Decoder部分的激活函数名称可选"relu,selu,leakyrelu" |
| --d_conv_dim  | 判别器的输入维度 |
| --gLayerNum  | 生成器的层数 |
| --resNum  | the number of resblock |
| --dLayerNum  | 判别器的层数 |
| --selected_attrs  | 选择的属性名字 |
| --total_step  | Totally training step |
| --batch_size  | Batch size |
| --g_lr  | Learning rate of generator |
| --d_lr  | Learning rate of discriminator |
| --lr_decay | Learning rate decay coefficient |
| --lr_decay_step | How many steps to performance the learning rate decay |
| --lr_decay_enable | 设置是否使用learning rate递减 |
| --D_step | 判别器训练代数 |
| --G_step | 生成器训练代数 |
| --GPWeight | GP权重系数 |
| --RecWeight | 重构权重系数 |
| --GAttrWeight | 生成器属性权重系数 |
| --DAttrWeight | 判别器属性权重系数 |
| --PSNR | 是否测试PSNR |
| --PSNRUseWhichAttr | 用哪一行属性测试PSNR |
| --PretrainedModelPath | 预训练模型根目录的绝对路径 |
| --use_pretrained_model | 是否使用预训模型 |
| --chechpoint_step | 使用的预训模型代数 |
| --cuda  | Set GPU device number |
| --parallel  | Enable the parallel training |
| --GPUs  | 如果使用多卡机，此处设置使用的GPU卡号用逗号隔开 |
| --dataset  | 选择数据名称，可选数据lsun,CelebA,cifar10,CelebAHQ，默认CelebAHQ |
| --specifiedImages  | 用来在训练时测试的图片名称 |
| --specifiedTestImages  | 用来测试的图片名称 |
| --logRootPath  | 日志存放的根目录，默认`./TrainingLogs`，checkpoint存在`./TrainingLogs/[version]/checkpoint/`，summary存放在`./TrainingLogs/[version]/summary/`，训练图片放在`./TrainingLogs/[version]/sample/`中，模型日志放在`./TrainingLogs/[version]/`根目录 |
| --testRoot  | 测试结果存放的路径，默认存放在`./TestFiles/[version]/` |
| --log_step  | 输出日志的代数 |
| --sample_step  | 输出采样图片的代数 |
| --model_save_step  | 模型存放的代数 |

# dataset path setting
The config  file is a json file, it is placed in `./dataTool/dataPath.json`

# Results



# Acknowledgement
- [STGAN](https://github.com/csmliu/STGAN)
- [STGAN-PyTorch](https://github.com/bluestyle97/STGAN-pytorch)