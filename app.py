import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple

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

_CREDENTIALS_PATH = Path("credentials.json")
credentials = json.load(_CREDENTIALS_PATH.open())
CREDENTIALS_DIR = _CREDENTIALS_PATH.resolve().parent
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


def _get_config_value(key: str, default=None):
    value = credentials.get(key)
    if value in ("", None):
        value = os.getenv(key, None)
    if value in ("", None):
        return default
    return value


def _resolve_config_path(raw_path: Optional[str]) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = (CREDENTIALS_DIR / path).resolve()
    return path


TTS_BASE_URL = _get_config_value("TTS_BASE_URL")
INDEXTTS_API_URL = _get_config_value("INDEXTTS_API_URL")
TTS_ENDPOINT_URL = None
if INDEXTTS_API_URL:
    normalized = INDEXTTS_API_URL.rstrip("/")
    if not normalized.lower().endswith("/tts"):
        normalized = normalized + "/tts"
    TTS_ENDPOINT_URL = normalized
elif TTS_BASE_URL:
    TTS_ENDPOINT_URL = TTS_BASE_URL.rstrip("/") + "/tts"

raw_timeout = _get_config_value("INDEXTTS_TIMEOUT")
try:
    INDEXTTS_TIMEOUT = float(raw_timeout) if raw_timeout not in (None, "") else None
except (TypeError, ValueError):
    INDEXTTS_TIMEOUT = None

DEFAULT_SPEAKER_AUDIO_PATH = _resolve_config_path(
    _get_config_value("DEFAULT_SPEAKER_AUDIO_PATH")
    or _get_config_value("INDEXTTS_PROMPT_WAV")
)
DEFAULT_EMO_AUDIO_PATH = _resolve_config_path(
    _get_config_value("DEFAULT_EMO_AUDIO_PATH")
)

INDEXTTS_DEFAULT_EMOTION = str(
    _get_config_value("INDEXTTS_EMOTION", "neutral") or "neutral"
).strip().lower()

_CHINESE_TEXT_PATTERN = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF。，、！？；：“”（）《》〈〉—…·\s]"
)


def _collect_common_tts_form_data() -> Dict[str, object]:
    value_specs = {
        "INDEXTTS_USE_RANDOM": lambda v: str(v).strip().lower() in {"true", "1", "yes", "y"},
        "INDEXTTS_INTERVAL_SILENCE": int,
        "INDEXTTS_MAX_TEXT_TOKENS": int,
        "INDEXTTS_TEMPERATURE": float,
        "INDEXTTS_TOP_P": float,
        "INDEXTTS_TOP_K": int,
        "INDEXTTS_REPETITION_PENALTY": float,
        "INDEXTTS_MAX_MEL_TOKENS": int,
    }
    data: Dict[str, object] = {}
    for key, caster in value_specs.items():
        raw_value = _get_config_value(key)
        if raw_value in (None, ""):
            continue
        try:
            data_key = key.replace("INDEXTTS_", "").lower()
            data[data_key] = caster(raw_value)
        except (TypeError, ValueError):
            continue
    return data


INDEXTTS_COMMON_FORM_DATA = _collect_common_tts_form_data()


def _collect_emotion_configs() -> Dict[str, Dict[str, object]]:
    raw_keys = set(credentials.keys()) | {
        key for key in os.environ.keys() if key.startswith("INDEXTTS_")
    }
    configs: Dict[str, Dict[str, object]] = {}
    prefix_map = {
        "INDEXTTS_EMO_WAV_": "emo_audio_path",
        "INDEXTTS_EMO_TEXT_": "emo_text",
        "INDEXTTS_EMO_VECTOR_JSON_": "emo_vector_json",
        "INDEXTTS_USE_EMO_TEXT_": "use_emo_text",
        "INDEXTTS_EMO_ALPHA_": "emo_alpha",
    }

    for prefix, field_name in prefix_map.items():
        for key in raw_keys:
            if not key.startswith(prefix):
                continue
            emotion_key = key[len(prefix) :].lower()
            value = _get_config_value(key)
            if value in (None, ""):
                continue
            entry = configs.setdefault(emotion_key, {})
            if field_name == "emo_audio_path":
                entry[field_name] = _resolve_config_path(value)
            elif field_name == "emo_alpha":
                try:
                    entry[field_name] = float(value)
                except (TypeError, ValueError):
                    continue
            elif field_name == "use_emo_text":
                entry[field_name] = str(value).strip().lower() in {"true", "1", "yes", "y"}
            else:
                entry[field_name] = value

    return configs


INDEXTTS_EMOTION_CONFIGS = _collect_emotion_configs()

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


def _load_audio_file(path: Path, error_detail: str) -> Tuple[str, bytes, str]:
    if not path or not path.exists():
        raise HTTPException(status_code=500, detail=error_detail)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=error_detail) from exc
    return (path.name, data, "audio/wav")


def _load_text_reference(value: str, error_detail: str) -> str:
    resolved_path = _resolve_config_path(value)
    if resolved_path and resolved_path.exists():
        try:
            return resolved_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise HTTPException(status_code=500, detail=error_detail) from exc
    return value


def _serialize_form_data(data: Dict[str, object]) -> Dict[str, str]:
    serialized: Dict[str, str] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, bool):
            serialized[key] = "true" if value else "false"
        else:
            serialized[key] = str(value)
    return serialized


def _filter_chinese_text(text: str) -> str:
    if not text:
        return ""

    filtered_chars = [char for char in text if _CHINESE_TEXT_PATTERN.fullmatch(char)]
    if not filtered_chars:
        return ""

    filtered_text = "".join(filtered_chars)
    return " ".join(filtered_text.split())


async def _forward_tts_request(
    text: str,
    speaker_file: Tuple[str, bytes, str],
    emo_file: Optional[Tuple[str, bytes, str]] = None,
    extra_form_data: Optional[Dict[str, object]] = None,
) -> Response:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text for voiceover cannot be empty.")

    if not TTS_ENDPOINT_URL:
        raise HTTPException(status_code=503, detail="TTS service is not configured.")

    files = {"speaker_audio": speaker_file}
    if emo_file is not None:
        files["emo_audio"] = emo_file

    form_data: Dict[str, object] = {"text": text}
    if INDEXTTS_COMMON_FORM_DATA:
        form_data.update(INDEXTTS_COMMON_FORM_DATA)
    if extra_form_data:
        form_data.update(extra_form_data)

    timeout = None
    if INDEXTTS_TIMEOUT is not None:
        timeout = httpx.Timeout(INDEXTTS_TIMEOUT)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                TTS_ENDPOINT_URL,
                data=_serialize_form_data(form_data),
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

    speaker_tuple = _load_audio_file(
        DEFAULT_SPEAKER_AUDIO_PATH,
        "Default speaker audio file was not found on the server.",
    )

    emotion_key = INDEXTTS_DEFAULT_EMOTION or ""
    emotion_config = INDEXTTS_EMOTION_CONFIGS.get(emotion_key)

    emo_tuple: Optional[Tuple[str, bytes, str]] = None
    extra_form_data: Dict[str, object] = {}

    if emotion_config:
        emo_audio_path = emotion_config.get("emo_audio_path")
        if isinstance(emo_audio_path, Path):
            emo_tuple = _load_audio_file(
                emo_audio_path,
                "Configured emotion audio file was not found.",
            )

        emo_text_value = emotion_config.get("emo_text")
        if isinstance(emo_text_value, str):
            extra_form_data["emo_text"] = _load_text_reference(
                emo_text_value,
                "Failed to load configured emotion text reference.",
            )

        emo_vector_value = emotion_config.get("emo_vector_json")
        if isinstance(emo_vector_value, str):
            extra_form_data["emo_vector_json"] = _load_text_reference(
                emo_vector_value,
                "Failed to load configured emotion vector reference.",
            )

        use_emo_text = emotion_config.get("use_emo_text")
        if use_emo_text is not None:
            extra_form_data["use_emo_text"] = use_emo_text

        emo_alpha = emotion_config.get("emo_alpha")
        if emo_alpha is not None:
            extra_form_data["emo_alpha"] = emo_alpha
    elif DEFAULT_EMO_AUDIO_PATH:
        emo_tuple = _load_audio_file(
            DEFAULT_EMO_AUDIO_PATH,
            "Configured default emotion audio file was not found.",
        )

    filtered_text = _filter_chinese_text(payload.text)
    if not filtered_text:
        raise HTTPException(
            status_code=400,
            detail="Chinese narration text is required for automatic voiceover.",
        )

    return await _forward_tts_request(filtered_text, speaker_tuple, emo_tuple, extra_form_data)

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
