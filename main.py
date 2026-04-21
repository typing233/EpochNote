from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import aiofiles
import httpx
import json
import base64
import re
import math
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import uuid

app = FastAPI(title="EpochNote - 智能科研管理工具 v2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

DATABASE_URL = "sqlite:///./epochnote.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TimelineNodeDB(Base):
    __tablename__ = "timeline_nodes"
    
    id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    label = Column(String, default="other")
    tags = Column(JSON, default=list)
    content = Column(Text, default="")
    file_name = Column(String, nullable=True)
    file_type = Column(String, nullable=True)
    file_path = Column(String, nullable=True)
    embedding_vector = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

AVAILABLE_TAGS = [
    "参数探索",
    "对照实验", 
    "预实验",
    "优化调整",
    "验证实验",
    "结果分析",
    "结论总结"
]

FILE_TYPE_MAPPING = {
    'image': ['png', 'jpg', 'jpeg', 'gif', 'svg', 'bmp', 'webp'],
    'pdf': ['pdf'],
    'code': ['py', 'js', 'ts', 'java', 'cpp', 'c', 'go', 'rs', 'html', 'css', 'json', 'yaml', 'yml'],
    'markdown': ['md', 'markdown'],
    'text': ['txt', 'log'],
    'audio': ['mp3', 'wav', 'ogg', 'flac', 'aac'],
    'video': ['mp4', 'webm', 'avi', 'mov', 'mkv']
}

class TimelineNode(BaseModel):
    id: str
    timestamp: str
    label: str
    tags: List[str] = []
    content: str
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    embedding_vector: Optional[List[float]] = None

class GenerateRequest(BaseModel):
    api_key: str
    timeline: List[TimelineNode]

class ExportRequest(BaseModel):
    content: str
    filename: Optional[str] = "experiment_narrative"

class AutoTagRequest(BaseModel):
    api_key: str
    content: str
    file_name: Optional[str] = None

class EmbeddingRequest(BaseModel):
    api_key: str
    content: str
    node_id: str

class SimilaritySearchRequest(BaseModel):
    api_key: Optional[str] = None
    node_id: str
    all_nodes: List[TimelineNode] = []
    top_k: int = 5

class UpdateTagsRequest(BaseModel):
    node_id: str
    tags: List[str]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_file_type(file_ext: str) -> str:
    ext = file_ext.lower().lstrip('.')
    for file_type, extensions in FILE_TYPE_MAPPING.items():
        if ext in extensions:
            return file_type
    return 'other'

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), timestamp: Optional[str] = None):
    try:
        file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        safe_filename = os.path.basename(file.filename) if file.filename else "unnamed_file"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        counter = 1
        while os.path.exists(file_path):
            name, ext = os.path.splitext(safe_filename)
            file_path = os.path.join(UPLOAD_DIR, f"{name}_{counter}{ext}")
            counter += 1
        
        content = await file.read()
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        file_type = get_file_type(file_ext)
        
        preview_content = ""
        if file_type in ["text", "code", "markdown"]:
            try:
                preview_content = content.decode('utf-8', errors='ignore')[:5000]
            except:
                preview_content = f"[二进制文件: {file.filename}]"
        else:
            preview_content = f"[{file_type.upper()} 文件: {file.filename}]"
        
        node_id = str(uuid.uuid4())
        timestamp_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        db = next(get_db())
        db_node = TimelineNodeDB(
            id=node_id,
            timestamp=timestamp_dt,
            label='other',
            tags=[],
            content=preview_content,
            file_name=os.path.basename(file_path),
            file_type=file_type,
            file_path=file_path,
            embedding_vector=[]
        )
        db.add(db_node)
        db.commit()
        db.refresh(db_node)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "id": node_id,
                "file_name": os.path.basename(file_path),
                "file_type": file_type,
                "timestamp": timestamp,
                "file_path": file_path,
                "content_preview": preview_content,
                "tags": [],
                "embedding_vector": []
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auto-tag")
async def auto_tag(request: AutoTagRequest):
    try:
        content_preview = request.content[:1000] if request.content else ""
        file_context = f"文件名: {request.file_name}\n" if request.file_name else ""
        
        prompt = f"""你是一位专业的科研助手。请根据以下实验记录内容，从给定的标签列表中选择最合适的2-3个标签。

给定标签列表（只能从中选择）：
{', '.join(AVAILABLE_TAGS)}

实验记录内容：
{file_context}
{content_preview}

请以JSON格式输出结果，格式如下：
{{
    "tags": ["标签1", "标签2"]
}}

要求：
1. 只能从给定的标签列表中选择
2. 选择2-3个最合适的标签
3. 直接输出JSON，不要有其他文字"""

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {request.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "你是一位专业的科研助手，擅长为实验记录分类打标签。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"DeepSeek API 调用失败: {response.text}"
                )
            
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            
            json_match = re.search(r'\{[\s\S]*\}', ai_response)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    tags = parsed.get("tags", [])
                except:
                    tags = []
            else:
                tags = []
            
            valid_tags = [tag for tag in tags if tag in AVAILABLE_TAGS]
            if not valid_tags:
                valid_tags = ["参数探索"]
            
            return JSONResponse(content={
                "success": True,
                "data": {
                    "tags": valid_tags[:3]
                }
            })
            
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"API 请求错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update-tags")
async def update_tags(request: UpdateTagsRequest):
    try:
        db = next(get_db())
        node = db.query(TimelineNodeDB).filter(TimelineNodeDB.id == request.node_id).first()
        
        if not node:
            raise HTTPException(status_code=404, detail="节点不存在")
        
        valid_tags = [tag for tag in request.tags if tag in AVAILABLE_TAGS]
        node.tags = valid_tags
        db.commit()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "node_id": request.node_id,
                "tags": valid_tags
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-embedding")
async def generate_embedding(request: EmbeddingRequest):
    try:
        content_for_embedding = request.content[:2000] if request.content else ""
        
        if not content_for_embedding or not content_for_embedding.strip():
            return JSONResponse(content={
                "success": True,
                "data": {
                    "node_id": request.node_id,
                    "embedding_vector": [],
                    "message": "内容为空，跳过向量生成"
                }
            })
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.deepseek.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {request.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "input": content_for_embedding
                }
            )
            
            if response.status_code != 200:
                return JSONResponse(content={
                    "success": False,
                    "data": {
                        "node_id": request.node_id,
                        "embedding_vector": [],
                        "error": f"API 调用失败: {response.text}"
                    }
                })
            
            result = response.json()
            
            if "data" not in result or not result["data"]:
                return JSONResponse(content={
                    "success": False,
                    "data": {
                        "node_id": request.node_id,
                        "embedding_vector": [],
                        "error": "API 返回格式不正确"
                    }
                })
            
            embedding = result["data"][0]["embedding"]
            
            db = next(get_db())
            node = db.query(TimelineNodeDB).filter(TimelineNodeDB.id == request.node_id).first()
            
            if node:
                node.embedding_vector = embedding
                db.commit()
            
            return JSONResponse(content={
                "success": True,
                "data": {
                    "node_id": request.node_id,
                    "embedding_vector": embedding
                }
            })
            
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "data": {
                "node_id": request.node_id,
                "embedding_vector": [],
                "error": str(e)
            }
        })

def keyword_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    
    text1 = text1.lower()
    text2 = text2.lower()
    
    keywords = {
        "参数探索": ["参数", "学习率", "超参数", "调整", "探索", "尝试"],
        "对照实验": ["对照", "对照组", "对比", "基准", "控制变量"],
        "预实验": ["预实验", "初步", "初探", "小规模", "测试"],
        "优化调整": ["优化", "改进", "调整", "策略", "调度器", "正则化"],
        "验证实验": ["验证", "确认", "检验", "验证集", "测试集"],
        "结果分析": ["分析", "结果", "混淆矩阵", "准确率", "错误率"],
        "结论总结": ["结论", "总结", "发现", "建议", "最终"]
    }
    
    score = 0.0
    max_score = 0.0
    
    for tag, words in keywords.items():
        count1 = sum(1 for word in words if word in text1)
        count2 = sum(1 for word in words if word in text2)
        max_score += 1.0
        if count1 > 0 and count2 > 0:
            score += 1.0
    
    text1_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text1))
    text2_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text2))
    
    if text1_words and text2_words:
        intersection = text1_words.intersection(text2_words)
        union = text1_words.union(text2_words)
        jaccard = len(intersection) / len(union) if union else 0
        score += jaccard * 2
        max_score += 2
    
    return score / max_score if max_score > 0 else 0.0

@app.post("/api/similarity-search")
async def similarity_search(request: SimilaritySearchRequest):
    try:
        if not request.all_nodes:
            db = next(get_db())
            target_node_db = db.query(TimelineNodeDB).filter(TimelineNodeDB.id == request.node_id).first()
            
            if not target_node_db:
                return JSONResponse(content={
                    "success": True,
                    "data": {
                        "similar_nodes": [],
                        "message": "目标节点不存在"
                    }
                })
            
            all_db_nodes = db.query(TimelineNodeDB).filter(
                TimelineNodeDB.id != request.node_id
            ).all()
            
            similarities = []
            target_embedding = target_node_db.embedding_vector
            target_content = target_node_db.content or ""
            
            for node in all_db_nodes:
                sim = 0.0
                
                if target_embedding and node.embedding_vector:
                    sim = cosine_similarity(target_embedding, node.embedding_vector)
                
                if sim <= 0:
                    sim = keyword_similarity(target_content, node.content or "")
                
                if sim > 0:
                    similarities.append({
                        "node_id": node.id,
                        "timestamp": node.timestamp.isoformat() if node.timestamp else None,
                        "label": node.label,
                        "tags": node.tags or [],
                        "content": node.content[:200] if node.content else "",
                        "file_name": node.file_name,
                        "file_type": node.file_type,
                        "similarity": sim
                    })
        else:
            target_node = None
            for node in request.all_nodes:
                if node.id == request.node_id:
                    target_node = node
                    break
            
            if not target_node:
                return JSONResponse(content={
                    "success": True,
                    "data": {
                        "similar_nodes": [],
                        "message": "目标节点不存在"
                    }
                })
            
            similarities = []
            target_embedding = target_node.embedding_vector or []
            target_content = target_node.content or ""
            
            for node in request.all_nodes:
                if node.id == request.node_id:
                    continue
                
                sim = 0.0
                
                if target_embedding and node.embedding_vector:
                    sim = cosine_similarity(target_embedding, node.embedding_vector)
                
                if sim <= 0:
                    sim = keyword_similarity(target_content, node.content or "")
                
                if sim > 0:
                    similarities.append({
                        "node_id": node.id,
                        "timestamp": node.timestamp,
                        "label": node.label,
                        "tags": node.tags or [],
                        "content": node.content[:200] if node.content else "",
                        "file_name": node.file_name,
                        "file_type": node.file_type,
                        "similarity": sim
                    })
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_k = similarities[:request.top_k]
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "similar_nodes": top_k
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "success": False,
            "data": {
                "similar_nodes": [],
                "error": str(e)
            }
        })

@app.get("/api/file-preview/{filename}")
async def get_file_preview(filename: str):
    try:
        filepath = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        file_ext = os.path.splitext(filename)[1].lower()
        file_type = get_file_type(file_ext)
        
        if file_type == "image":
            with open(filepath, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            mime_type = "image/png"
            if file_ext in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
            elif file_ext == ".gif":
                mime_type = "image/gif"
            elif file_ext == ".svg":
                mime_type = "image/svg+xml"
            elif file_ext == ".webp":
                mime_type = "image/webp"
            
            return JSONResponse(content={
                "success": True,
                "data": {
                    "file_type": "image",
                    "mime_type": mime_type,
                    "content": f"data:{mime_type};base64,{image_data}"
                }
            })
        
        elif file_type in ["text", "code", "markdown"]:
            async with aiofiles.open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
            
            return JSONResponse(content={
                "success": True,
                "data": {
                    "file_type": file_type,
                    "content": content
                }
            })
        
        elif file_type == "pdf":
            with open(filepath, "rb") as f:
                pdf_data = base64.b64encode(f.read()).decode('utf-8')
            
            return JSONResponse(content={
                "success": True,
                "data": {
                    "file_type": "pdf",
                    "content": f"data:application/pdf;base64,{pdf_data}"
                }
            })
        
        elif file_type in ["audio", "video"]:
            mime_type = "audio/mpeg"
            if file_type == "video":
                mime_type = "video/mp4"
                if file_ext == ".webm":
                    mime_type = "video/webm"
            else:
                if file_ext == ".wav":
                    mime_type = "audio/wav"
                elif file_ext == ".ogg":
                    mime_type = "audio/ogg"
            
            return JSONResponse(content={
                "success": True,
                "data": {
                    "file_type": file_type,
                    "mime_type": mime_type,
                    "file_url": f"/uploads/{filename}"
                }
            })
        
        else:
            return JSONResponse(content={
                "success": True,
                "data": {
                    "file_type": "other",
                    "content": f"文件类型 {file_ext} 暂不支持在线预览"
                }
            })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate_narrative(request: GenerateRequest):
    try:
        sorted_timeline = sorted(request.timeline, key=lambda x: x.timestamp)
        
        timeline_text = ""
        for i, node in enumerate(sorted_timeline, 1):
            tags_str = ", ".join(node.tags) if node.tags else "无"
            timeline_text += f"""
【节点 {i}】
时间: {node.timestamp}
标签: {node.label} (自动标签: {tags_str})
内容: {node.content}
文件: {node.file_name or '无'}
---
"""
        
        prompt = f"""你是一位经验丰富的科研工作者。请根据以下实验时间线记录，生成一段逻辑连贯、专业的实验叙事文本。

要求：
1. 按照时间顺序组织内容，体现实验的发展脉络
2. 使用学术化但易懂的语言
3. 强调关键决策和实验结果的关联性
4. 输出纯文本，不要使用Markdown格式
5. 字数控制在1500-3000字之间

实验时间线：
{timeline_text}

请生成实验叙事文本："""

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {request.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "你是一位专业的科研助手，擅长将碎片化的实验记录整理成逻辑连贯的实验叙事和论文方法部分。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4000
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"DeepSeek API 调用失败: {response.text}"
                )
            
            result = response.json()
            narrative = result["choices"][0]["message"]["content"]
            
            return JSONResponse(content={
                "success": True,
                "data": {
                    "narrative": narrative,
                    "generated_at": datetime.now().isoformat()
                }
            })
            
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"API 请求错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export/markdown")
async def export_markdown(request: ExportRequest):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{request.filename}_{timestamp}.md"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        markdown_content = f"""# 实验叙事记录

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 实验内容

{request.content}

---

*本文档由 EpochNote 智能科研管理工具生成 v2.0*
"""
        
        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(markdown_content)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "filename": filename,
                "filepath": filepath
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        filepath,
        media_type="application/octet-stream",
        filename=filename
    )

@app.get("/api/test-data")
async def get_test_data():
    test_nodes = [
        {
            "id": "test_1",
            "timestamp": "2024-01-15T09:00:00",
            "label": "param-explore",
            "tags": ["参数探索", "预实验"],
            "content": "开始探索新的超参数组合。尝试将学习率从 0.001 调整为 0.01，观察模型收敛速度。初始结果显示损失函数下降明显加快，但可能存在过拟合风险。需要进一步监控验证集准确率。",
            "file_name": None,
            "file_type": None
        },
        {
            "id": "test_2",
            "timestamp": "2024-01-15T14:30:00",
            "label": "control",
            "tags": ["对照实验", "参数探索"],
            "content": "设置对照组实验。使用原始学习率 0.001 作为基准，与新学习率 0.01 进行对比。实验设计：两组实验仅学习率不同，其他条件保持一致（批次大小 32，优化器 Adam， dropout 0.2）。",
            "file_name": "experiment_config.py",
            "file_type": "code"
        },
        {
            "id": "test_3",
            "timestamp": "2024-01-16T10:00:00",
            "label": "preliminary",
            "tags": ["预实验", "结果分析"],
            "content": "预实验结果分析。学习率 0.01 的实验组在训练集上准确率达到 98%，但验证集仅 85%，存在明显过拟合。对照组学习率 0.001 的训练集准确率 92%，验证集 89%，泛化能力更好。结论：需要引入学习率衰减策略。",
            "file_name": "training_curve.png",
            "file_type": "image"
        },
        {
            "id": "test_4",
            "timestamp": "2024-01-17T09:00:00",
            "label": "optimization",
            "tags": ["优化调整", "参数探索"],
            "content": "实施优化调整。引入余弦退火学习率调度器，初始学习率设为 0.005，每 10 个 epoch 衰减一次。同时增加 L2 正则化项（权重 0.0001）以防止过拟合。修改了模型配置文件，添加了新的调度器实现。",
            "file_name": "model_config.yaml",
            "file_type": "code"
        },
        {
            "id": "test_5",
            "timestamp": "2024-01-18T16:00:00",
            "label": "verification",
            "tags": ["验证实验", "结果分析"],
            "content": "验证实验结果。使用优化后的配置重新训练模型。训练集准确率 95%，验证集准确率 92%，测试集准确率 91%。过拟合问题得到显著改善。对比之前的基准模型，验证集准确率提升了 3 个百分点。",
            "file_name": "validation_report.md",
            "file_type": "markdown"
        },
        {
            "id": "test_6",
            "timestamp": "2024-01-19T11:00:00",
            "label": "analysis",
            "tags": ["结果分析", "结论总结"],
            "content": "深度结果分析。分析各模型的混淆矩阵：\n1. 原始模型：对类别3的分类错误率较高（15%）\n2. 优化后模型：类别3的错误率降至 8%\n3. 主要改进来自学习率调度策略，正则化贡献较小\n\n统计显著性检验（t检验）显示 p-value = 0.023，改进具有统计学意义。",
            "file_name": "confusion_matrix.png",
            "file_type": "image"
        },
        {
            "id": "test_7",
            "timestamp": "2024-01-20T14:00:00",
            "label": "conclusion",
            "tags": ["结论总结"],
            "content": "实验结论总结。\n\n主要发现：\n1. 学习率对模型性能有显著影响，过高的学习率会导致过拟合\n2. 余弦退火学习率调度器能够有效平衡收敛速度和泛化能力\n3. 最佳配置：初始学习率 0.005 + 余弦退火 + L2 正则化\n\n建议：\n- 未来可以探索学习率预热（warmup）策略\n- 尝试在更大的数据集上验证这些发现\n- 考虑结合早停（early stopping）机制",
            "file_name": "conclusion.pdf",
            "file_type": "pdf"
        }
    ]
    
    return JSONResponse(content={
        "success": True,
        "data": {
            "nodes": test_nodes,
            "available_tags": AVAILABLE_TAGS,
            "description": "这是一组用于测试 v2.0 新功能的示例数据。包含了完整的实验流程：参数探索、对照实验、预实验、优化调整、验证实验、结果分析和结论总结。"
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=2567, reload=True)
