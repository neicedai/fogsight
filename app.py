import asyncio
import json
import mimetypes
import os
import re
import shutil
import tempfile
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
from playwright.async_api import Page, async_playwright
from starlette.background import BackgroundTask
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

DEFAULT_EXPORT_DURATION = 8.0
MIN_EXPORT_DURATION = 3.0
MAX_EXPORT_DURATION = 120.0
EXPORT_BUFFER_SECONDS = 0.5

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


def _cleanup_directory(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _coerce_positive_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if value > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            numeric = float(stripped)
            return numeric if numeric > 0 else None
        except ValueError:
            match = re.search(r"(\d+(?:\.\d+)?)", stripped)
            if match:
                try:
                    numeric = float(match.group(1))
                    return numeric if numeric > 0 else None
                except ValueError:
                    return None
    return None


async def _probe_stream_duration(path: Path, stream_selector: str) -> Optional[float]:
    if shutil.which("ffprobe") is None:
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        stream_selector,
        "-show_entries",
        "stream=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return None

    stdout, _ = await process.communicate()
    if process.returncode != 0 or not stdout:
        return None

    try:
        duration = float(stdout.decode().strip().splitlines()[0])
        return duration if duration > 0 else None
    except (ValueError, IndexError):
        return None


async def _probe_audio_duration(path: Path) -> Optional[float]:
    return await _probe_stream_duration(path, "a:0")


async def _probe_video_duration(path: Path) -> Optional[float]:
    return await _probe_stream_duration(path, "v:0")


async def _extract_animation_duration(page: Page) -> Optional[float]:
    candidate_scripts = [
        "window.fogsightAnimationDuration",
        "window.animationDuration",
        "window.__FOGSIGHT_ANIMATION_DURATION__",
        "window.__ANIMATION_DURATION__",
        "document.body?.dataset?.animationDuration",
    ]

    for script in candidate_scripts:
        try:
            value = await page.evaluate(f"() => {{ try {{ return {script}; }} catch (e) {{ return null; }} }}")
        except Exception:
            continue
        duration = _coerce_positive_float(value)
        if duration:
            return duration

    try:
        dataset_value = await page.evaluate(
            "() => {"
            "  const el = document.querySelector('[data-animation-duration]');"
            "  return el ? el.dataset.animationDuration || el.getAttribute('data-animation-duration') : null;"
            "}"
        )
    except Exception:
        dataset_value = None

    duration = _coerce_positive_float(dataset_value)
    if duration:
        return duration

    try:
        meta_value = await page.evaluate(
            "() => {"
            "  const meta = document.querySelector('meta[name=\"animation-duration\"], meta[name=\"video-duration\"], meta[name=\"fogsight:animation-duration\"]');"
            "  return meta ? meta.content : null;"
            "}"
        )
    except Exception:
        meta_value = None

    duration = _coerce_positive_float(meta_value)
    if duration:
        return duration

    try:
        gsap_duration = await page.evaluate(
            "() => {"
            "  const timeline = typeof gsap !== 'undefined' ? (gsap.globalTimeline || gsap.timeline?.()) : null;"
            "  if (!timeline) return null;"
            "  if (typeof timeline.totalDuration === 'function') return timeline.totalDuration();"
            "  if (typeof timeline.duration === 'function') return timeline.duration();"
            "  if (typeof timeline.totalTime === 'function') return timeline.totalTime();"
            "  return null;"
            "}"
        )
    except Exception:
        gsap_duration = None

    duration = _coerce_positive_float(gsap_duration)
    if duration:
        return duration

    return None


async def _determine_render_duration(page: Page, audio_path: Optional[Path]) -> float:
    durations: List[float] = []

    page_duration = await _extract_animation_duration(page)
    if page_duration:
        durations.append(page_duration)

    if audio_path is not None and audio_path.exists():
        audio_duration = await _probe_audio_duration(audio_path)
        if audio_duration:
            durations.append(audio_duration)

    if not durations:
        return DEFAULT_EXPORT_DURATION

    longest = max(durations)
    buffered = longest + EXPORT_BUFFER_SECONDS
    constrained = max(MIN_EXPORT_DURATION, min(MAX_EXPORT_DURATION, buffered))
    return constrained


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


@app.post("/export")
async def export_animation(
    html: str = Form(...),
    audio: Optional[UploadFile] = File(None),
):
    """Render the generated animation and stream it back as an MP4 file."""

    if not html or not html.strip():
        raise HTTPException(status_code=400, detail="Animation HTML is required for export.")

    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=500, detail="ffmpeg is not available on the server.")

    temp_dir_path = Path(tempfile.mkdtemp(prefix="export_"))
    audio_path: Optional[Path] = None
    recorded_video_path: Optional[Path] = None
    output_path = temp_dir_path / "animation.mp4"

    try:
        (temp_dir_path / "animation.html").write_text(html, encoding="utf-8")

        if audio is not None:
            audio_bytes = await audio.read()
            await audio.close()
            if audio_bytes:
                suffix = Path(audio.filename or "").suffix
                if not suffix and audio.content_type:
                    guessed = mimetypes.guess_extension(audio.content_type)
                    if guessed:
                        suffix = guessed
                if not suffix:
                    suffix = ".wav"
                audio_path = temp_dir_path / f"voiceover{suffix}"
                audio_path.write_bytes(audio_bytes)

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            context = None
            page = None
            video = None
            try:
                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    record_video_dir=str(temp_dir_path),
                    record_video_size={"width": 1920, "height": 1080},
                )
                page = await context.new_page()
                await page.set_content(html, wait_until="networkidle")
                await page.wait_for_timeout(500)

                render_duration = await _determine_render_duration(page, audio_path)
                wait_ms = max(0, int(render_duration * 1000))
                if wait_ms:
                    await page.wait_for_timeout(wait_ms)

                video = page.video
            finally:
                if context is not None:
                    await context.close()
                elif page is not None:
                    try:
                        await page.close()
                    except Exception:
                        pass
                await browser.close()

        if video is None:
            raise HTTPException(status_code=500, detail="Failed to capture animation video.")

        try:
            recorded_video_path = Path(await video.path())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save recorded video: {exc}") from exc

        if recorded_video_path is None or not recorded_video_path.exists():
            raise HTTPException(status_code=500, detail="Recorded animation video was not found.")

        if output_path.exists():
            output_path.unlink()

        if audio_path is not None and not audio_path.exists():
            audio_path = None

        video_duration = await _probe_video_duration(recorded_video_path)
        audio_duration: Optional[float] = None

        if audio_path is not None:
            audio_duration = await _probe_audio_duration(audio_path)
            audio_filters: List[str] = []
            include_shortest = False
            duration_tolerance = 0.05

            if video_duration is not None and audio_duration is not None:
                if audio_duration + duration_tolerance < video_duration:
                    audio_filters.append("apad")
                elif audio_duration > video_duration + duration_tolerance:
                    include_shortest = True
            else:
                # When durations cannot be determined, prefer padding to avoid
                # prematurely ending the muxed output.
                audio_filters.append("apad")

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(recorded_video_path),
                "-i",
                str(audio_path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]

            if audio_filters:
                ffmpeg_cmd.extend(["-af", ",".join(audio_filters)])

            ffmpeg_cmd.extend([
                "-movflags",
                "+faststart",
            ])

            if include_shortest:
                ffmpeg_cmd.append("-shortest")

            ffmpeg_cmd.append(str(output_path))
        else:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(recorded_video_path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]

        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()

        if process.returncode != 0 or not output_path.exists():
            detail = stderr.decode(errors="ignore") if stderr else "ffmpeg failed to render the video."
            raise HTTPException(status_code=500, detail=detail.strip() or "Video export failed.")

        def iterfile() -> Any:
            with output_path.open("rb") as file:
                while True:
                    chunk = file.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"animation-{timestamp}.mp4"
        headers = {"Content-Disposition": f"attachment; filename={filename}"}
        background = BackgroundTask(_cleanup_directory, str(temp_dir_path))

        return StreamingResponse(iterfile(), media_type="video/mp4", headers=headers, background=background)

    except HTTPException:
        _cleanup_directory(str(temp_dir_path))
        raise
    except Exception as exc:
        _cleanup_directory(str(temp_dir_path))
        raise HTTPException(status_code=500, detail=f"Failed to export video: {exc}") from exc

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
