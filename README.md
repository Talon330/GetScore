# getscore

用于爬取川大硕士研究生考试成绩, 需要自行准备考生姓名和考生编号, 利用此项目[cnn_captcha](https://github.com/nickliqian/cnn_captcha)识别验证码, 验证码比较简单, 准确率挺高的, 能达到90%以上. 

或许明年就不能用了...

## 1.目录结构

### 1.1 文件夹介绍

| 文件夹名称            | 说明                    |
| ---------------- |:--------------------- |
| `./cnnlib/`      | 封装CNN的相关代码目录          |
| `./GET_CAPTCHA/` | 存储识别后的验证码图片,文件名包含识别结果 |
| `./model/`       | 存放模型文件                |

### 1.2文件介绍

| 文件名                   | 说明                                    |
| --------------------- | ------------------------------------- |
| `downloadValicode.py` | 下载验证码图片丰富训练集                          |
| `split.py`            | 分割验证码为单个字符                            |
| `getscore.py`         | 获取成绩并保存                               |
| `复试名单.csv`            | 复试名单需要自行准备,按给出的模板填入相应内容即可,其中姓名和考生编号必填 |

### 1.3运行后新建文件/文件夹介绍

| 默认名称            | 说明                                                                        |
| --------------- | ------------------------------------------------------------------------- |
| `./image/`      | 存贮`downloadValicode.py`下载的验证码图片, 文件名要手动修改为该图片所显示的字符, 也是`split.py`中待分割图片路径 |
| `./denoised/`   | `split.py`去噪后的训练集图片存放路径                                                   |
| `./denoserr/`   | `split.py`分割出错时图片转存地址                                                     |
| `./singlechar/` | `split.py`完成验证码分割后,单个字符图片存放位置                                             |
| `查询失败的信息.csv`   | 仅当查询失败次数过多才会出现                                                            |
| `temporary.jpg` | 运行过程中暂存图片,结束时删除                                                           |

## 2  如何使用

### 2.1 依赖

```
pip install -r requirements.txt
```

### 2.2 训练模型

验证码识别使用的是[cnn_captcha](https://github.com/nickliqian/cnn_captcha), 若有需要,利用`downloadValicode.py`下载文件标注(也可直接用`GET_CAPTCHA/`文件夹内的图片训练, 标注信息自行检查正确与否),按其说明重新训练即可.

* 提供训练好的模型文件, 存储于`./model/`,可酌情使用.

### 2.3 查询数据准备

按格式把待查询信息填入`复试名单.csv`即可.

### 2.4 获取成绩

打开终端, 进入文件所在目录, 运行:

```
python3 getscore.py
```

* 勿疯狂下载成绩, 给服务器歇口气, 出什么问题自行负责！
* 如有侵权，联系立即删除