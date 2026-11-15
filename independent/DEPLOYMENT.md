# 在未安装豆包输入法的 Mac 上部署

可以将这个部署目录复制到另一台兼容的 Mac。运行需要完整的 `independent/exported-model/`、本项目运行代码和 Python 依赖；不需要豆包安装、账户、原 FLUTE 包或私有推理库。

已验证环境是 Apple Silicon、macOS 26.2、Python 3.13、MLX 0.32.4。Intel Mac 和其他 macOS / Python 版本未验证。这里“离线”指依赖准备好后的录音与识别；首次准备 Python 和安装依赖另算。不要直接复制 `.venv` 到另一台机器，虚拟环境可能包含原机器绝对路径。

## 生成部署目录（源 Mac）

在源码仓库根目录执行：

```sh
independent/.venv/bin/python independent/package.py \
  --model independent/exported-model \
  --output /path/to/doubao-mlx-portable
```

部署目录只包含：运行必需的 Python 文件、录音网页、依赖清单、沙箱配置、文档，以及转换后的模型。不会复制输入法、私有库、FLUTE 解析器、提取器、`.venv` 或原 `.model` 文件。默认拒绝覆盖已有目录；输出含忽略全部内容的 `.gitignore`，权重和派生资产仅在用户本机保留。

模型目录可包含全部九个子模型，但当前运行代码只使用非流式的三个。**完整导出流式权重，不等于已经实现流式推理。**

## 目标 Mac 可联网安装依赖

将整个部署目录拷到目标 Mac 后，进入该目录：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r independent/requirements.txt
.venv/bin/python independent/web_server.py
```

打开 <http://localhost:8766>，授权麦克风并开始录音。建议 5–15 秒，页面 17 秒停止，后端最多 18 秒。录音后处理全部在本机进行，不需要联网。命令行：

```sh
.venv/bin/python independent/transcribe.py independent/exported-model input-16k.wav
```

## 目标 Mac 从安装到识别都不能联网

先在架构、macOS、Python 版本兼容的联网 Mac 上，为部署目录下载 wheel：

```sh
python3 -m pip download --only-binary=:all: \
  -r independent/requirements.txt -d wheels
```

把 `wheels/` 一起拷走。目标 Mac 仍需事先安装兼容的 Python；本部署目录不打包 Python 或 macOS 系统框架。在目标目录创建新的虚拟环境并离线安装：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links wheels \
  -r independent/requirements.txt
.venv/bin/python independent/web_server.py
```

wheel 文件受平台、系统与 Python 版本约束。不要把当前 Apple Silicon / Python 3.13 的 wheel 当成适用于任意 Mac 的通用安装包。

## 2026-11-15 同机隔离实测

已复制完整九组件模型和运行代码到源项目以外的新目录，重新创建不继承系统 site-packages 的虚拟环境。离线依赖包使用 Python 3.13 / Apple Silicon 的兼容 wheel；MLX wheel 标记要求 macOS 26.0 或更新。没有复制原 `.venv`。

- 在禁止网络访问的沙箱中，`pip --no-index --find-links wheels` 完成依赖安装。
- 禁止读取整个原项目、旧运行目录、豆包安装/数据目录及所有 `.model` 文件；实际尝试读取这些文件均得到权限拒绝，网络连接也被拒绝。
- 校验完整导出目录全部文件哈希，通过；运行目录不含 `weights.py`、`extract.py` 或私有库。
- 三段命令行测试通过：开会样本 47 个 token、雨伞样本 35 个 token、静音 0 个 token；最终文字与既有结果一致。单次含加载耗时分别约 3.82、2.17、1.27 秒，不作为性能基准。

- 独立端口的录音网页加载及音频 POST 后端测试通过；测试输入为已有合成音频，不是新录制的真人麦克风样本。

## 验证边界

同机隔离部署能验证是否暗中依赖源项目、豆包安装、原模型包或现有虚拟环境；不能替代在另一台物理 Mac 上的兼容性和麦克风测试。实际验证结果记录在源码仓库的 README。目标机器首次打开页面仍需要浏览器麦克风权限。

模型参数及派生资产仍来自豆包；复制格式不改变许可或再分发权限。本项目公开实现代码，不公开权重。
