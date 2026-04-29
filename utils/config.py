
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
# override=True 表示如果系统环境变量中有同名变量，也强制使用 .env 里的（可选）
load_dotenv(dotenv_path="api_key.env", override=True)

# --- 基础路径 ---
PERSIST_PATH = "data/chroma_db"

# --- 高德地图 API 配置 ---
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

# --- 智谱 AI 配置 ---
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# --- 代理/Gemini 配置 ---
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
PROXY_ENDPOINT = "http://127.0.0.1:7897"

# --- 阿里云 API 配置 ---
ALI_API_KEY = os.getenv("ALI_API_KEY")
ALI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# --- 模型列表 ---
MODEL_LIST = [
    "glm-4.5-air",
    "glm-4-flash",
    "gemini-3-flash",
    "gemini-3.1-pro",
    "claude-opus-4-6-thinking",
    "qwen-turbo",
    "qwen-plus",
    "qwen-max",
    "qwen-vl-max",
    "qwen-audio-turbo",

]

# --- 初始化环境 (移除代理) ---
def init_env():
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)
    os.environ.pop('ALL_PROXY', None)