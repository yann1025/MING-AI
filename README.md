# 🏥 MING 医疗对话助手

基于 **Qwen1.5-1.8B-Chat** + **MING-MOE 医疗 LoRA** 的中文医疗问答助手，支持流式输出 + Gradio Web 界面。

> ⚠️ 本助手仅供学习参考，**不能替代专业医生诊断**。

## ✨ 效果预览

<img width="1911" height="912" alt="image" src="https://github.com/user-attachments/assets/8e616c92-a603-4fbe-97f8-efd5c554434c" />
<img width="1591" height="918" alt="image" src="https://github.com/user-attachments/assets/99d67f07-d984-4bd2-9b42-088cf916583e" />


## 🛠 硬件要求

- **显卡**：NVIDIA GPU，显存 ≥ 4GB（实测 RTX 4050 6GB 完美运行）
- **内存**：≥ 8GB
- **系统**：Windows / Linux

## 📦 环境安装

### 1. 创建 conda 环境
\`\`\`bash
conda create -n ming python=3.10 -y
conda activate ming
\`\`\`

### 2. 安装 PyTorch (CUDA 11.8)
\`\`\`bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 -f https://mirrors.aliyun.com/pytorch-wheels/cu118/
\`\`\`

### 3. 安装其他依赖
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## 📥 下载模型

### 基座模型 Qwen1.5-1.8B-Chat
从 [HF Mirror](https://hf-mirror.com/Qwen/Qwen1.5-1.8B-Chat) 或 [ModelScope](https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat) 下载，放到 `models/Qwen1.5-1.8B-Chat/`

### MING 医疗 LoRA
从 [HF Mirror](https://hf-mirror.com/BlueZeros/MING-MOE-1.8B) 下载，放到 `models/MING-MOE-1.8B/`

最终目录结构：
\`\`\`
medicine-MING/
├── models/
│   ├── Qwen1.5-1.8B-Chat/         # 基座模型
│   └── MING-MOE-1.8B/             # 医疗 LoRA
├── start_med_chat.py              # 启动脚本
├── requirements.txt
└── README.md
\`\`\`

## 🚀 运行

修改 `start_med_chat.py` 中的模型路径后：

\`\`\`bash
python start_med_chat.py
\`\`\`

浏览器自动打开 `http://127.0.0.1:7860` 即可对话。

## 💡 致谢

- 基座模型：[Qwen1.5-1.8B-Chat](https://github.com/QwenLM/Qwen1.5)
- 医疗 LoRA：[MING](https://github.com/MediaBrain-SJTU/MING)

## 📄 License

仅供学习交流使用。
