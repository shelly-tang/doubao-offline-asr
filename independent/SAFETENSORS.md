# 从本机权重导出独立 safetensors 模型

导出器将已安装版本中的非流式编码器、主解码器和共享 embedding/logit 权重，连同运行必需的配置，转换成独立目录。转换完成后，不需要 `asr_model.model`、`plain.model`、输入法安装目录或私有推理库。默认导出过程只读原安装文件，临时 plaintext pack 会自动清理，不删除使用者原来的文件。

## 导出与运行

从仓库根目录执行：

```sh
independent/.venv/bin/python -m pip install -r independent/requirements.txt
independent/.venv/bin/python independent/export.py
independent/.venv/bin/python independent/transcribe.py independent/exported-model input-16k.wav
independent/.venv/bin/python independent/web_server.py
```

首次使用请先按照主 README 创建虚拟环境并安装受支持版本的输入法和离线模型。纯转换无需 Metal GPU；推理仍需要支持 MLX 的 Apple Silicon 环境。

如果已有本地 plaintext pack，可以跳过原安装提取：

```sh
independent/.venv/bin/python independent/export.py --pack independent/plain.model
```

自定义目标目录（已存在时拒绝覆盖）：

```sh
independent/.venv/bin/python independent/export.py --output /path/to/my-model
independent/.venv/bin/python independent/web_server.py --model /path/to/my-model
```

`--pack` 仍作为网页 `--model` 参数的兼容别名。命令行和缓存检查均可接受模型目录或原 plaintext pack；目录加载失败时不会自动回退到原包。网页原有 18 秒长度限制不变。

## 目录内容与加载原理

| 文件 | 用途 |
|---|---|
| `encoder.safetensors` | 793 个张量，卷积前端、24 层 Conformer 和音频投影 |
| `decoder.safetensors` | 517 个张量，28 层主解码器及归一化参数 |
| `embedding.safetensors` | 4 个张量，共享 token embedding / logit 权重等 |
| `*.graph.json` | 节点输入名、属性等现有 Python 实现所需图配置 |
| `model.json` | 格式版本、量化布局标记、组件/资产索引、文件 SHA-256 与大小 |
| `vocab.txt`、`merges.txt`、`add_special_tokens.txt` | 本地 BPE 词表及特殊 token |
| `config.yaml` | 本地提示模板等原配置；不执行其中的私有 actor 实现 |
| `cmvn.txt`、`fbank.conf` | 音频前端资产 |
| `.gitignore` | 忽略该目录全部内容，避免误上传权重及衍生资产 |

本次导出总大小约 **470.8 MB**（约 449 MiB）。只导出当前非流式运行路径；流式模型、草稿模型和 VAD 未导出。模型配置中保留的 `.pthlite` 名称只是逻辑索引，不会打开这些文件。

每个权重张量保留原 dtype、shape 和字节；INT4 依然是带符号 nibble 打包的 INT8 数组，并保存原缩放系数。没有重新量化，也没有把存储膨胀为 FP32。加载后继续由现有 MLX 实现解包为 FP32 计算，与转换前算法相同。

`model_source.py` 使用 `safetensors.numpy.load_file` 直接加载权重，首次读取每个文件时检查大小与 SHA-256。导出目录路径不调用 FLUTE 解析器。JSON 是数据，不加载 pickle 或执行序列化代码。哈希用于检测损坏，不是来源签名。

这是**本项目格式的模型目录**，不能直接当成 Transformers / MLX-LM 的通用 Qwen checkpoint。参数和派生配置仍来自豆包；转换不改变权重许可。仓库不会发布这些生成文件。

## 转换验证

导出时自动回读每个 safetensors，对全部 1,314 个张量检查名称、dtype、shape、逐字节内容；任何差异会中止。先在临时目录完成所有文件，最后才生成目标目录，避免中断留下半成品。

可复现的实际音频对比：

```sh
independent/.venv/bin/python independent/verify_export.py \
  --pack independent/plain.model \
  --model independent/exported-model \
  first-16k.wav second-16k.wav silence-16k.wav
```

此命令为了建立对照，需要保留或另行提取旧 plaintext pack；正常识别不需要。每个样本分别启动原包路径和导出目录路径，比较全部生成 token ID、原始文字、最终文字、截断状态以及音频编码器输出。导出目录路径额外通过 macOS 沙箱禁止读取所有 `.model` 文件；两条路径都禁止网络及豆包私有库访问。

2026-11-15 本机结果：

| 样本 | token 数 | 识别输出 | 编码器最大误差 | 原包 / 导出目录耗时 |
|---|---:|---|---:|---|
| 约 5.92 秒合成语音 | 47 | 你好，今天下午3点开会，请帮我准备15份会议材料。 | 0 | 4.46 / 3.01 秒 |
| 约 6.10 秒合成语音 | 35 | 明天早上8:30，请提醒我带上雨伞，坐地铁去图书馆。 | 0 | 2.57 / 2.82 秒 |
| 两秒静音 | 0 | 空文本 | 0 | 1.81 / 1.86 秒 |

三项的 token ID、原始文字和最终文字全部一致，且没有截断。耗时包含各进程权重加载，仅作观测记录；进程按顺序执行，受缓存与系统负载影响，不构成速度提升结论。此验证比较的是转换前后的独立 MLX 实现，不是与豆包原库的逐层对齐。

不需要真实权重或 GPU 的格式测试：

```sh
independent/.venv/bin/python -m unittest discover -s independent -p 'test_*.py' -v
```

五项测试覆盖标量/INT8 无损加载、不调用原包接口、损坏或缺失文件、未知版本与目录外路径拒绝。真人语音准确率、长音频和原库数值对齐仍保留原实验版限制。
