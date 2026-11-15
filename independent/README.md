# 独立 MLX 推理实验版

这个目录直接读取原版权重，用 NumPy + MLX 实现音频到文字的非流式推理。**运行时不加载豆包输入法、`runtime.bundle`、`helper.dylib` 或私有 Frameworks。** 它不是根目录旧版私有运行库的另一层包装。

目前完成的是短音频端到端原型。两段中文合成语音已正确识别，两秒静音输出为空；这不代表真人、噪声、长音频和逐算子数值一致性已经通过验证。

## 安装与使用

先克隆本仓库，在仓库根目录执行。需要 Apple Silicon Mac、可用的 Metal GPU 和 Python 3.13；本次验证环境为 macOS 26.2、MLX 0.32.4，其他系统版本尚未验证。首次安装 Python 依赖需要联网，提取和推理不下载任何远程模型。

```sh
python3 -m venv independent/.venv
independent/.venv/bin/python -m pip install -r independent/requirements.txt
independent/.venv/bin/python independent/extract.py
```

提取脚本只读取本机已安装的豆包程序和模型文件，不执行其中的代码。它检查程序与模型的 SHA-256，只支持此前分析的版本（模型目录 `1_8`）。输出 `independent/plain.model`，约 825 MB；如果已存在会拒绝覆盖。提取后，推理只需要这个文件和本目录代码、Python 依赖，不再读取原始安装目录。

录音界面：

```sh
independent/.venv/bin/python independent/web_server.py
```

打开 <http://localhost:8766>，点击开始录音，再停止识别。建议 5–15 秒；页面 17 秒自动停止，后端最多接受 18 秒。麦克风必须由用户授权并主动开始录音。界面仍使用原项目的录音组件，但识别后端全部换成独立 MLX。没有接入 VAD 自动断句，也没有边说边出字。

命令行（单声道、16 kHz、PCM16 WAV）：

```sh
independent/.venv/bin/python independent/transcribe.py independent/plain.model input.wav
```

其他采样率可先用 macOS 自带工具转换：

```sh
afconvert -f WAVE -d LEI16@16000 -c 1 input.wav input-16k.wav
```

JSON 中 `text` 是处理数字标注后的结果，`raw_text` 保留模型原始输出，`truncated` 表示是否触及生成长度上限。没有加载网络客户端或远程模型服务。网页只监听本机地址，临时录音请求结束后删除。

## 如何确认没有借用私有运行库

网页推理子进程默认通过 `sandbox-exec` 启动，禁用网络、读取输入法安装和用户数据目录、读取 `runtime.bundle` / `helper.dylib` / 根目录私有 `Frameworks`。

也可以手动验证命令行：

```sh
sandbox-exec \
  -D DOUBAO_DATA="$HOME/Library/Application Support/DoubaoIme" \
  -D PRIVATE_FRAMEWORKS="$PWD/Frameworks" \
  -f independent/offline.sb \
  independent/.venv/bin/python independent/transcribe.py independent/plain.model input-16k.wav
```

MLX 自己及 macOS 的 Metal 等系统库当然仍然参与运算；这里排除的是豆包的私有推理实现。模型提取阶段需要读取本机安装文件，推理阶段不需要。

## 实现与证据边界

| 部分 | 独立实现 | 验证情况 |
|---|---|---|
| 模型包 | FLUTE v1 目录与 Panther FlatBuffer 读取 | 九个子模型全部通过张量尺寸检查 |
| 提取 | 从安装文件读取 XOR 混淆参数并还原 payload | 输出与旧版 plaintext pack SHA-256 完全一致 |
| 4-bit 权重 | 每八个输入通道，前四个低 nibble、后四个高 nibble；有符号 INT4 | 对照本机二进制中保留的算子布局解析，独立编写 NumPy 解包 |
| 音频前端 | 16 kHz、25 ms 窗、10 ms 帧移、80-bin log-mel、CMVN | 当前按 Kaldi 风格重建，未完成原库逐帧对齐 |
| 编码器 | 两层卷积、24 层 Conformer、四帧拼接和音频投影 | 两段合成语音端到端正确；分块边界/量化注意力尚未逐层校验 |
| 解码器 | 28 层、RMSNorm、Q/K norm、GQA、RoPE、SiLU、KV cache | FP32 整段与逐 token 最大差值约 5.5e-5 |
| 词表与提示词 | 本地 BPE、原包 default prompt、音频嵌入插入 | 数字 ITN 标注可解析为阿拉伯数字 |
| 草稿解码器、流式、VAD | 未接入 | 当前直接用主解码器逐 token 生成 |

解码器结构与 Qwen3 类似，但这些结果仍不能证明权重来源是某个官方 Qwen3 checkpoint。

原库中存在 W4A8 等量化路径。本原型将 INT4 原权重解包成 FP32，默认使用浮点激活计算，**并非原版整数算子的逐 bit 复现**。因此内存比原包大，结果和速度可能与豆包原版不同。`decoder.py` 还保留了实验性的激活 INT8 路径，端到端入口没有启用它。

## 本机验证记录

- 首段：约 5.92 秒合成音频，输出“你好，今天下午3点开会，请帮我准备15份会议材料。”；初次端到端约 4.67 秒，含权重加载。
- 新生成的另一段：“明天早上八点半，请提醒我带上雨伞，坐地铁去图书馆。”输出正确，时间转换为 `8:30`；约 4.58 秒，含加载。运行时已禁止网络与豆包程序/私有库访问。
- 录音页面后端复测第二段（6.099 秒）：请求总等待 3.603 秒，其中模型加载 1.764 秒、其余处理 1.555 秒，差额为请求/进程启动开销。
- 两秒全零 PCM：输出空文本；约 1.92 秒，含加载。
- 这些是开发 smoke test，**不是准确率评测，也不是稳定性能基准**。真人语音仍需用户实际录音测试。

`smoke.py` 检查解码器有限输出与缓存一致性，不测试语音准确率：

```sh
independent/.venv/bin/python independent/smoke.py independent/plain.model --float-activations
```

## 限制

当前最多 18 秒，尚未验证长音频跨块状态；不保证与原版完全相同。没有系统输入框上下文、历史对话、改词记录采集；提示词中的这些字段填空。网页每次请求重新加载模型。后续需要逐层比较中间张量、补充真人与噪声样本、验证长音频，再考虑常驻模型和低精度优化。

公开内容只有实现代码。不提交权重、原二进制、私有库或其中的算子源码；生成的 `.model`、`.npy`、`.npz`、`.safetensors` 均已加入忽略规则。

## 文件说明

| 文件 | 用途 |
|---|---|
| `extract.py` | 只读提取本机权重；校验原文件哈希 |
| `weights.py` | 容器、图与张量解析，INT4 解包 |
| `features.py` / `encoder.py` | 音频特征与 Conformer 编码器 |
| `decoder.py` / `text.py` | 主解码器、BPE、提示词和数字标注处理 |
| `transcribe.py` | 短音频命令行入口，输出 JSON |
| `web_server.py` / `offline.sb` | 本地录音后端及推理沙箱 |
| `smoke.py` | 解码器缓存一致性检查 |

## 是否可以不再保留豆包模型包

提取后运行不再需要原始下载的 `asr_model.model` 或原安装目录，但仍需要本地 `plain.model`。可进一步把张量和配置导出为 safetensors 等标准格式，让推理也脱离 FLUTE 容器；当前尚未实现该导出器。这是格式转换，不是重新训练，不改变参数来源或再分发权限。要完全摆脱豆包参数，需要替换为许可合适的模型或自行训练。
