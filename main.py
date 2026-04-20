from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import aiofiles
import httpx
import json
from datetime import datetime

app = FastAPI(title="EpochNote - 智能科研管理工具")

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

app.mount("/static", StaticFiles(directory="static"), name="static")

class TimelineNode(BaseModel):
    id: str
    timestamp: str
    label: str
    content: str
    file_name: Optional[str] = None
    file_type: Optional[str] = None

class GenerateRequest(BaseModel):
    api_key: str
    timeline: List[TimelineNode]

class ExportRequest(BaseModel):
    content: str
    filename: Optional[str] = "experiment_narrative"

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
        
        file_type = "text"
        if file_ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']:
            file_type = "code"
        elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp']:
            file_type = "image"
        elif file_ext in ['.txt', '.md', '.log']:
            file_type = "text"
        
        preview_content = ""
        if file_type in ["text", "code"]:
            try:
                preview_content = content.decode('utf-8', errors='ignore')[:5000]
            except:
                preview_content = f"[二进制文件: {file.filename}]"
        else:
            preview_content = f"[{file_type.upper()} 文件: {file.filename}]"
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "id": os.path.basename(file_path),
                "file_name": safe_filename,
                "file_type": file_type,
                "timestamp": timestamp,
                "file_path": file_path,
                "content_preview": preview_content
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
            timeline_text += f"""
【节点 {i}】
时间: {node.timestamp}
标签: {node.label}
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

*本文档由 EpochNote 智能科研管理工具生成*
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=2567, reload=True)
