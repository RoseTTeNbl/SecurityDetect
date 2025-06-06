# 智能家庭安防系统

基于深度学习的智能家庭安防系统，集成了人脸识别、火灾检测等功能。

## 1. 环境部署

### 1.1 系统要求
- Python 3.8+
- CUDA 11.0+ (如果使用GPU)
- Windows 10/11

### 1.2 安装依赖
```bash
# 创建虚拟环境 
conda create -n sct python==3.9
conda activate sct # Windows

#安装torch
pip install torch==2.1.0 torchvision==0.16.0
#or cuda(先下载到本地)
pip install cu121\torch-2.1.0+cu121-cp39-cp39-win_amd64.whl
pip install cu121\torchvision-0.16.0+cu121-cp39-cp39-win_amd64.whl

# 安装依赖
pip install -r requirements.txt
```

### 1.3 项目结构
```
智能家庭安防系统/
├── main.py              # 主程序
├── myui.py             # UI界面定义
├── face_recognition.py # 人脸识别模块
├── train.py           # 模型训练脚本
├── requirements.txt   # 依赖列表
├── settings.ini       # 配置文件
├── fire_alarm.mp3    # 火灾警报音频
├── stranger_alarm.mp3 # 陌生人警报音频
├── start_screen.jpg  # 启动界面图片
├── logs/             # 日志目录
├── users/            # 用户人脸图片目录
└── model/           
    ├── best.pt/      #火灾烟雾检测模型   
```


## 2. UI操作手册

### 2.1 主界面功能
- **打开摄像头**：启动/关闭摄像头
- **开始检测**：开启/关闭检测功能
- **开启警报**：启用/禁用警报声音
- **开始录制**：开始/停止视频录制

### 2.2 设置区域
- **录制目录**：设置视频录制保存位置
- **截图目录**：设置陌生人截图保存位置
- **火灾检测阈值**：调节火灾检测的灵敏度（0.6-0.7推荐）
- **人脸识别阈值**：调节人脸识别的灵敏度（0.6-0.9推荐）

### 2.3 警报指示灯
- **火灾**：检测到火灾时亮红灯
- **烟雾**：检测到烟雾时亮红灯
- **陌生人**：检测到未知人员时亮红灯

### 2.4 状态信息
- 实时显示检测结果和系统状态
- 所有事件都会记录到logs目录下的日志文件中

## 3. 软件设计文档

### 3.1 系统架构
系统采用模块化设计，主要包含以下组件：
1. **UI模块** (myui.py)
   - 基于PyQt5实现
   - 提供用户交互界面
   - 实时显示摄像头画面和状态信息

2. **人脸识别模块** (face_recognition.py)
   - 使用MTCNN进行人脸检测
   - 使用FaceNet提取人脸特征
   - 实现人脸比对和身份识别

3. **火灾检测模块**
   - 基于YOLOv8实现
   - 支持火灾和烟雾检测
   - 实时分析视频流

4. **主控模块** (main.py)
   - 协调各个功能模块
   - 处理用户输入
   - 管理系统状态
   - 实现警报和录制功能

### 3.2 数据流
1. 摄像头捕获画面 → 图像预处理
2. 并行进行人脸识别和火灾检测
3. 结果整合并更新UI显示
4. 触发相应的警报和记录机制

## 4. 自定义训练指南

### 4.1 添加新用户人脸
1. 在`users`目录下创建以用户名命名的文件夹
2. 将用户的人脸照片（多角度）放入该文件夹
3. 运行人脸数据库构建：
```bash
python build_face_database.py
```

### 4.2 训练火灾检测模型
提供的预训练模型(`fire_smoke_detection\yolov8n_training\weights\best.pt`)是基于yolov8n微调的，如果用户有特定场景检测的需要可以自己重新微调模型。
1. 准备数据集：
   - 在`dataset/images`中放入训练图片
   - 在`dataset/labels`中放入对应的标注文件
   
2. 运行训练脚本：
```bash
python train_alarm.py
```

训练参数可在train.py中调整：


## 5. 打包为exe

使用PyInstaller打包项目：

```bash
# 安装PyInstaller
pip install pyinstaller

# 打包命令
pyinstaller --name="智能家庭安防系统" ^
            --windowed ^
            --icon=icon.ico ^
            --add-data "fire_alarm.mp3;." ^
            --add-data "stranger_alarm.mp3;." ^
            --add-data "start_screen.jpg;." ^
            --add-data "fire_smoke_detection/yolov8n_training/weights/best.pt;fire_smoke_detection/yolov8n_training/weights/" ^
            main.py
```

注意事项：
1. 确保所有资源文件都已添加到打包列表
2. 检查路径分隔符是否正确（Windows使用分号，Linux使用冒号）
3. 打包后的程序在dist目录下
4. 首次运行可能需要创建相关目录

## 6. 常见问题

1. **摄像头打不开**
   - 检查设备管理器中摄像头是否正常
   - 尝试更换摄像头索引（0或1）
   - 建议安装iVCam用于远程连接摄像头，支持电脑直接调用手机摄像头。安装地址https://www.e2esoft.com/ivcam/

2. **人脸识别不准**
   - 调整人脸识别阈值
   - 添加更多角度的人脸照片
   - 确保光线充足

3. **火灾检测误报**
   - 调整火灾检测阈值
   - 重新训练模型，增加数据集多样性

4. **程序闪退**
   - 检查日志文件排查错误
   - 确保所有依赖正确安装
   - 验证模型文件是否完整

## 7. 联系与支持

如有问题，请通过以下方式获取帮助：
1. 查看日志文件（logs目录）
2. 提交Issue
3. 发送邮件至[您的邮箱]

## 8. 许可证

本项目采用MIT许可证。详见LICENSE文件。 