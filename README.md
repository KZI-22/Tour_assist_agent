# Travel Assistant

一个基于 `Streamlit + LangGraph + LangChain + Chroma` 的旅游助手项目，支持旅游问答、行程规划、多模态输入和本地知识库检索。

## 功能特性

- 多轮旅游对话
- 旅游行程自动规划
- 景点、美食、天气、路线查询
- 图片景点识别
- 语音转文字
- 本地文档入库与 RAG 检索
- 节点执行过程可视化

## 项目结构

```text
AGENT_TOUR_copy/
├─ agents/              # Router / Researcher / Planner 工作流节点
├─ core/                # LLM、工具、知识库管理
├─ data/                # Chroma 持久化数据
├─ utils/               # 配置与环境变量
├─ main.py              # Streamlit 入口
├─ UI.py                # UI 组件与样式
├─ api_key.env          # 本地密钥配置（不要提交）
├─ requirements.txt     # 依赖列表
└─ README.md
```

## 运行环境

- Python 3.10 或 3.11

安装依赖：

```bash
pip install -r requirements.txt
```

## 环境变量

在项目根目录准备 `api_key.env`：

```env
AMAP_API_KEY=你的高德地图Key
ZHIPU_API_KEY=你的智谱Key
PROXY_API_KEY=你的代理或 Gemini Key
ALI_API_KEY=你的阿里百炼Key
```

说明：

- `AMAP_API_KEY`：天气、景点、美食、路线查询
- `ZHIPU_API_KEY`：GLM 模型与 embedding
- `PROXY_API_KEY`：Gemini / 代理模型
- `ALI_API_KEY`：Qwen 多模态和音频能力

## 启动项目

```bash
streamlit run main.py
```

## 支持的知识库文件

- `pdf`
- `txt`
- `docx`
- `csv`

上传后会自动切分、去重并写入 `data/chroma_db/`。


