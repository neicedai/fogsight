import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import pytz
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
try:
    import google.generativeai as genai
except ModuleNotFoundError:
    from google import genai
# -----------------------------------------------------------------------
# 0. 配置
# -----------------------------------------------------------------------
shanghai_tz = pytz.timezone("Asia/Shanghai")

credentials = json.load(open("credentials.json"))
APP_ROOT = Path(__file__).resolve().parent
API_KEY = credentials["API_KEY"]
BASE_URL = credentials.get("BASE_URL", "")
MODEL = credentials.get("MODEL", "gemini-2.5-pro")

if API_KEY.startswith("sk-"):
    # 为 OpenRouter 添加应用标识
    extra_headers = {}
    if "openrouter.ai" in BASE_URL.lower():
        extra_headers = {
            "HTTP-Referer": "https://github.com/fogsightai/fogsight",
            "X-Title": "Fogsight - AI Animation Generator"
        }
    
    client = AsyncOpenAI(
        api_key=API_KEY, 
        base_url=BASE_URL,
        default_headers=extra_headers
    )
    USE_GEMINI = False
else:
    os.environ["GEMINI_API_KEY"] = API_KEY
    gemini_client = genai.Client()
    USE_GEMINI = True

if API_KEY.startswith("sk-REPLACE_ME"):
    raise RuntimeError("请在环境变量里配置 API_KEY")

TTS_BASE_URL = credentials.get("TTS_BASE_URL") or os.getenv("TTS_BASE_URL")
TTS_ENDPOINT = credentials.get("TTS_ENDPOINT") or os.getenv("TTS_ENDPOINT")
TTS_DEFAULT_PARAMETERS: Dict[str, Any] = credentials.get("TTS_DEFAULT_PARAMETERS", {})
TTS_DEFAULT_EMOTION = credentials.get("TTS_DEFAULT_EMOTION") or os.getenv("TTS_DEFAULT_EMOTION")
TTS_EMOTION_PRESETS: Dict[str, Dict[str, Any]] = credentials.get("TTS_EMOTION_PRESETS", {})
DEFAULT_SPEAKER_AUDIO_PATH = (
    credentials.get("DEFAULT_SPEAKER_AUDIO_PATH")
    or os.getenv("DEFAULT_SPEAKER_AUDIO_PATH")
)
DEFAULT_EMO_AUDIO_PATH = (
    credentials.get("DEFAULT_EMO_AUDIO_PATH")
    or os.getenv("DEFAULT_EMO_AUDIO_PATH")
)

def _resolve_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = APP_ROOT / path
    return path


def _load_binary_file(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read {description} at {path}.",
        ) from exc


def _prepare_emotion_payload(
    emotion: Optional[str],
    emo_file: Optional[tuple],
) -> tuple[Dict[str, Any], Optional[tuple]]:
    if not emotion:
        return {}, emo_file

    preset = None
    key = emotion.lower()
    if key in TTS_EMOTION_PRESETS:
        preset = TTS_EMOTION_PRESETS[key]
    else:
        key = emotion.upper()
        preset = TTS_EMOTION_PRESETS.get(key)

    if not preset:
        return {}, emo_file

    data_updates: Dict[str, Any] = {}

    if "emo_text" in preset and preset["emo_text"]:
        data_updates["emo_text"] = preset["emo_text"]

    if "use_emo_text" in preset:
        data_updates["use_emo_text"] = preset["use_emo_text"]

    if "emo_alpha" in preset:
        data_updates["emo_alpha"] = preset["emo_alpha"]

    vector_value = preset.get("emo_vector_json")
    if vector_value:
        vector_path = _resolve_path(vector_value)
        if vector_path and vector_path.exists():
            try:
                data_updates["emo_vector_json"] = vector_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to read emotion vector JSON at {vector_path}.",
                ) from exc
        else:
            data_updates["emo_vector_json"] = vector_value

    if emo_file is None:
        wav_value = preset.get("emo_wav") or preset.get("emo_wav_path")
        if wav_value:
            wav_path = _resolve_path(wav_value)
            if wav_path and wav_path.exists():
                emo_bytes = _load_binary_file(wav_path, "emotion reference audio")
                emo_file = (
                    wav_path.name,
                    emo_bytes,
                    "audio/wav",
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Emotion reference audio was not found at {wav_value}",
                )

    return data_updates, emo_file

templates = Jinja2Templates(directory="templates")

# -----------------------------------------------------------------------
# 1. FastAPI 初始化
# -----------------------------------------------------------------------
app = FastAPI(title="AI Animation Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    topic: str
    history: Optional[List[dict]] = None


class AutoVoiceoverRequest(BaseModel):
    text: str
    topic: Optional[str] = None

# -----------------------------------------------------------------------
# 2. 核心：流式生成器 (现在会使用 history)
# -----------------------------------------------------------------------
async def llm_event_stream(
    topic: str,
    history: Optional[List[dict]] = None,
    model: str = None, # Will use MODEL from config if not specified
) -> AsyncGenerator[str, None]:
    history = history or []
    
    # Use configured model if not specified
    if model is None:
        model = MODEL
    
    # The system prompt is now more focused
    system_prompt = f"""请你生成一个非常精美的动态动画,讲讲 {topic}
要动态的,要像一个完整的,正在播放的视频。包含一个完整的过程，能把知识点讲清楚。
页面极为精美，好看，有设计感，同时能够很好的传达知识。知识和图像要准确
附带一些旁白式的文字解说,从头到尾讲清楚一个小的知识点
不需要任何互动按钮,直接开始播放
使用和谐好看，广泛采用的浅色配色方案，使用很多的，丰富的视觉元素。双语字幕
**请保证任何一个元素都在一个2k分辨率的容器中被摆在了正确的位置，避免穿模，字幕遮挡，图形位置错误等等问题影响正确的视觉传达**
html+css+js+svg，放进一个html里"""

    if USE_GEMINI:
        try:
            full_prompt = system_prompt + "\n\n" + topic
            if history:
                history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
                full_prompt = history_text + "\n\n" + full_prompt
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: gemini_client.models.generate_content(
                    model=model, 
                    contents=full_prompt
                )
            )
            
            text = response.text
            chunk_size = 50
            
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                payload = json.dumps({"token": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.05)
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": topic},
        ]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.8, 
            )
        except OpenAIError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            if token:
                payload = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.001)

    yield 'data: {"event":"[DONE]"}\n\n'

# -----------------------------------------------------------------------
# 3. 路由 (CHANGED: Now a POST request)
# -----------------------------------------------------------------------
@app.post("/generate")
async def generate(
    chat_request: ChatRequest, # CHANGED: Use the Pydantic model
    request: Request,
):
    """
    Main endpoint: POST /generate
    Accepts a JSON body with "topic" and optional "history".
    Returns an SSE stream.
    """
    accumulated_response = ""  # for caching flow results

    async def event_generator():
        nonlocal accumulated_response
        try:
            async for chunk in llm_event_stream(chat_request.topic, chat_request.history):
                accumulated_response += chunk
                if await request.is_disconnected():
                    break
                yield chunk
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


    async def wrapped_stream():
        async for chunk in event_generator():
            yield chunk

    headers = {
        "Cache-Control": "no-store",
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(wrapped_stream(), headers=headers)


def _get_tts_endpoint() -> str:
    endpoint = TTS_ENDPOINT or ""
    if endpoint:
        return endpoint.rstrip("/")

    if not TTS_BASE_URL:
        raise HTTPException(status_code=503, detail="TTS service is not configured.")

    base = TTS_BASE_URL.rstrip("/")
    if base.endswith("/tts"):
        return base
    return base + "/tts"


async def _forward_tts_request(
    text: str,
    speaker_file: tuple,
    emo_file: Optional[tuple] = None,
    emotion: Optional[str] = None,
) -> Response:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text for voiceover cannot be empty.")

    tts_endpoint = _get_tts_endpoint()

    data: Dict[str, Any] = {"text": text}
    if TTS_DEFAULT_PARAMETERS:
        data.update(TTS_DEFAULT_PARAMETERS)

    emotion_updates, emo_file = _prepare_emotion_payload(
        emotion or TTS_DEFAULT_EMOTION,
        emo_file,
    )
    if emotion_updates:
        data.update(emotion_updates)

    files = {"speaker_audio": speaker_file}
    if emo_file is not None:
        files["emo_audio"] = emo_file

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                tts_endpoint,
                data={k: v for k, v in data.items() if v is not None},
                files=files,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"TTS request failed: {exc}") from exc

    if response.status_code != 200:
        detail = response.text
        try:
            detail_json = response.json()
            detail = detail_json.get("detail") or detail_json.get("error") or detail
        except ValueError:
            pass
        raise HTTPException(status_code=response.status_code, detail=detail)

    headers = {
        "Content-Disposition": "attachment; filename=voiceover.wav",
        "Content-Type": response.headers.get("content-type", "audio/wav"),
    }
    return Response(content=response.content, headers=headers)


@app.post("/voiceover")
async def generate_voiceover(
    text: str = Form(...),
    speaker_audio: UploadFile = File(...),
    emo_audio: Optional[UploadFile] = File(None),
):
    """Proxy text-to-speech requests to the configured TTS service."""

    speaker_bytes = await speaker_audio.read()
    speaker_tuple = (
        speaker_audio.filename or "prompt.wav",
        speaker_bytes,
        speaker_audio.content_type or "audio/wav",
    )

    emo_tuple = None
    if emo_audio is not None:
        emo_bytes = await emo_audio.read()
        emo_tuple = (
            emo_audio.filename or "emotion.wav",
            emo_bytes,
            emo_audio.content_type or "audio/wav",
        )

    return await _forward_tts_request(text, speaker_tuple, emo_tuple)


@app.post("/voiceover/auto")
async def generate_auto_voiceover(payload: AutoVoiceoverRequest):
    """Generate voiceover using the default speaker audio without manual input."""

    if not DEFAULT_SPEAKER_AUDIO_PATH:
        raise HTTPException(
            status_code=503,
            detail="Default speaker audio is not configured for automatic voiceover.",
        )

    speaker_resolved = _resolve_path(DEFAULT_SPEAKER_AUDIO_PATH)
    if not speaker_resolved or not speaker_resolved.exists():
        raise HTTPException(
            status_code=500,
            detail="Default speaker audio file was not found on the server.",
        )

    speaker_path = speaker_resolved

    speaker_bytes = _load_binary_file(speaker_path, "default speaker audio")

    speaker_tuple = (
        speaker_path.name,
        speaker_bytes,
        "audio/wav",
    )

    emo_tuple = None
    if DEFAULT_EMO_AUDIO_PATH:
        emo_path = _resolve_path(DEFAULT_EMO_AUDIO_PATH)
        if not emo_path or not emo_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Configured default emotion audio file was not found.",
            )
        emo_bytes = _load_binary_file(emo_path, "default emotion audio")
        emo_tuple = (
            emo_path.name,
            emo_bytes,
            "audio/wav",
        )

    return await _forward_tts_request(
        payload.text,
        speaker_tuple,
        emo_tuple,
    )

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(
        "index.html", {
            "request": request,
            "time": datetime.now(shanghai_tz).strftime("%Y%m%d%H%M%S")})

# -----------------------------------------------------------------------
# 4. 本地启动命令
# -----------------------------------------------------------------------
# uvicorn app:app --reload --host 0.0.0.0 --port 8000


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
