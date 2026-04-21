import os
import shutil
import time
import gc
import hashlib
import re
import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# import config
# from llm_core import get_embeddings
from utils import config
from AGENT_TOUR.core.llm_core import get_embeddings


def _cleanup_persist_path():
    """尽力清理持久化目录，避免损坏数据长期残留。"""
    if not os.path.exists(config.PERSIST_PATH):
        return

    try:
        shutil.rmtree(config.PERSIST_PATH)
        return
    except Exception:
        pass

    for filename in os.listdir(config.PERSIST_PATH):
        file_path = os.path.join(config.PERSIST_PATH, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception:
            pass


def _normalize_text_for_chunking(text):
    """针对常见 OCR/PDF 提取噪声做轻量规范化。"""
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u3000", " ").replace("\xa0", " ")

    # 修复英文连字符断行，如 "inter-\nnational" -> "international"
    normalized = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", normalized)

    # 清理多余空格与过长空行
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    # 合并大多数软换行：保留段落换行(\n\n)，单换行尽量并入同一段
    normalized = re.sub(
        r"(?<!\n)(?<=[^\n。！？；：.!?;])\n(?=[^\n•\-*\d])",
        " ",
        normalized,
    )
    return normalized.strip()


def _filter_existing_chunk_ids(vector_db, chunk_map):
    """过滤已在向量库中的 chunk id，避免跨批次重复入库。"""
    if vector_db is None or not chunk_map:
        return chunk_map, 0

    candidate_ids = list(chunk_map.keys())
    existing_ids = set()

    try:
        existing = vector_db.get(ids=candidate_ids, include=[])
        existing_ids = set(existing.get("ids", []))
    except Exception:
        # 兼容不同版本封装
        try:
            existing = vector_db._collection.get(ids=candidate_ids, include=[])
            existing_ids = set(existing.get("ids", []))
        except Exception:
            existing_ids = set()

    if not existing_ids:
        return chunk_map, 0

    filtered_map = {chunk_id: doc for chunk_id, doc in chunk_map.items() if chunk_id not in existing_ids}
    return filtered_map, len(existing_ids)

# --- 加载数据库 ---
def load_db():
    if not os.path.exists(config.PERSIST_PATH):
        return None

    try:
        has_data = bool(os.listdir(config.PERSIST_PATH))
    except Exception:
        has_data = True

    if not has_data:
        return None

    try:
        vector_db = Chroma(
            persist_directory=config.PERSIST_PATH,
            embedding_function=get_embeddings(),
            collection_name="my_docs"
        )
        # 健康检查：损坏库通常会在这里抛错
        vector_db.get(limit=1, include=[])
        return vector_db
    except Exception as e:
        print(f"检测到数据库损坏，准备重建: {str(e)}")
        try:
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        _cleanup_persist_path()
    return None


def ingest_documents(uploaded_files, vector_db, selected_model):
    """
    uploaded_files: Streamlit UploadedFile 对象的列表 (List[UploadedFile])
    vector_db: 当前的 Chroma 实例 (可能为 None)
    selected_model: 当前选中的模型名
    returns: (updated_vector_db, success_message)
    """
    if not uploaded_files:
        return vector_db, {"success": False, "message": "没有检测到上传的文件。"}

    all_chunks = [] # 用于收集所有文件的文本块
    success_count = 0

    # 1. 遍历所有上传的文件
    for file in uploaded_files:
        temp_path = f"temp_{file.name}"
        
        # 将文件流写入本地临时文件
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        
        try:
            # 2. 根据文件后缀名动态选择 Loader
            ext = os.path.splitext(file.name)[-1].lower()
            
            if ext == ".pdf":
                loader = PyPDFLoader(temp_path)
            elif ext == ".txt":
                loader = TextLoader(temp_path, encoding="utf-8")
            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(temp_path)
            elif ext == ".csv":
                loader = CSVLoader(temp_path, encoding="utf-8")
            else:
                print(f"跳过不支持的文件格式: {file.name}")
                continue # 如果是不支持的格式，直接跳过处理下一个

            # 加载当前文件
            data = loader.load()
            
            # --- 核心改进：文档清洗与顶尖切分逻辑 ---
            # 1. 预清洗数据：解决 PDF 等格式常见的断词、软换行、多余空白问题
            for doc in data:
                doc.page_content = _normalize_text_for_chunking(doc.page_content)

                # 修正 source，避免临时文件名污染元数据
                if not isinstance(doc.metadata, dict):
                    doc.metadata = {}
                doc.metadata["source"] = file.name
            
            # 2. 针对中文环境与智谱 embedding-3 优化的切分策略
            # 智谱 embedding 模型对语义相关性非常敏感，因此切分第一原则是：绝对不能把一句话切断！
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,        # 稍作下调，兼顾检索精度与大模型的上下文利用效率
                chunk_overlap=150,     # 保持适当重叠，保证跨块信息不丢失
                length_function=len,
                # 设计切分优先级：双换行(段落) > 单换行 > 常见中文句首/句尾标点 > 空格
                separators=[
                    "\n\n", 
                    "\n", 
                    "。", 
                    "！", 
                    "？", 
                    "；", 
                    "”", 
                    "——", 
                    "，",  # 若句子实在太长，退而求其次在逗号处切断
                    " ", 
                    ""
                ],
                # 如果当前使用的 langchain_text_splitters 版本支持，该参数能保证切分不丢失原本的标点符号
                is_separator_regex=False
            )
            
            # 生成切分块
            chunks = text_splitter.split_documents(data)
            
            # 将当前文件的文本块追加到总列表中
            all_chunks.extend(chunks)
            success_count += 1
            
        except Exception as e:
            print(f"处理文件 {file.name} 时出错: {str(e)}")
            
        finally:
            # 3. 无论成功与否，务必清理本地临时文件，防止硬盘塞满
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 4. 统一批量入库
    if not all_chunks:
        return vector_db, {"success": False, "message": "未能从上传的文件中提取到有效文本！"}

    # 使用 MD5 为每个文本块生成唯一 ID，并进行批次内去重
    unique_chunks_dict = {}
    for chunk in all_chunks:
        # 以 chunk 的纯文本内容生成 MD5 哈希值，保证相同内容的块哈希一致
        content_hash = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()
        if content_hash not in unique_chunks_dict:
            unique_chunks_dict[content_hash] = chunk
            
    final_chunks = list(unique_chunks_dict.values())
    final_ids = list(unique_chunks_dict.keys())

    # 显式跨批次去重：入库前检查已有 ID，防止重复向量污染检索
    new_chunk_map, existed_count = _filter_existing_chunk_ids(vector_db, unique_chunks_dict)
    new_chunks = list(new_chunk_map.values())
    new_ids = list(new_chunk_map.keys())

    if not new_chunks:
        return vector_db, {"success": True, "message": (
            f"成功解析 {success_count} 个文件！"
            f"(共提取 {len(all_chunks)} 块，批次去重后 {len(final_chunks)} 块，"
            f"其中 {existed_count} 块已存在于知识库，本次未新增)"
        )}

    if vector_db is not None:
        vector_db.add_documents(documents=new_chunks, ids=new_ids)
    else:
        vector_db = Chroma.from_documents(
            documents=new_chunks,
            embedding=get_embeddings(), 
            ids=new_ids,
            persist_directory=config.PERSIST_PATH, 
            collection_name="my_docs"
        )
    
    return vector_db, {"success": True, "message": (
        f"成功解析并入库 {success_count} 个文件！"
        f"(共提取 {len(all_chunks)} 块，批次去重后 {len(final_chunks)} 块，"
        f"跨批次过滤已存在 {existed_count} 块，实际新增 {len(new_chunks)} 块)"
    )}

# --- 彻底清空数据库 ---
def clear_database(vector_db):
    """
    处理复杂的 Chroma 关闭与文件删除逻辑
    """
    if vector_db is not None:
        client = vector_db._client
        
        # 逻辑删除
        try:
            vector_db.delete_collection()
        except Exception:
            pass
            
        # 停止系统线程
        try:
            client._system.stop()
        except Exception:
            pass
            
        # 清理缓存
        try:
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass
            
        gc.collect()
        time.sleep(0.5)

    # 物理删除
    if os.path.exists(config.PERSIST_PATH):
        try:
            shutil.rmtree(config.PERSIST_PATH)
            return True, "数据库及文件夹已彻底物理删除！"
        except Exception as e:
            # 兜底：掏空文件夹
            for filename in os.listdir(config.PERSIST_PATH):
                file_path = os.path.join(config.PERSIST_PATH, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception:
                    pass
            return True, "数据已彻底清空！(残留空壳将在重启后解锁)"
    
    return False, "数据库不存在"