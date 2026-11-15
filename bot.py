import asyncio
import html
import json
import logging
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, RetryAfter
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ==============================================================================
# 1. KONFIGURASI PATH RELATIF (AMAN UNTUK PTERODACTYL / WISPBYTE & TERMUX)
# ==============================================================================
# Menggunakan lokasi absolut direktori script agar file .env dan .db tidak
# pernah tersasar meskipun working directory container Pterodactyl berubah.
BASE_DIR: Path = Path(__file__).resolve().parent
ENV_PATH: Path = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("DualBackendBot")

TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

_raw_db_path = os.getenv("DB_PATH", "bot_memory.db").strip()
DB_PATH: Path = (
    Path(_raw_db_path)
    if Path(_raw_db_path).is_absolute()
    else (BASE_DIR / _raw_db_path)
)

SYSTEM_PROMPT: str = os.getenv(
    "SYSTEM_PROMPT",
    "Kamu adalah asisten AI yang cerdas, ramah, akurat, dan menjawab dengan jelas.",
).strip()
MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
PROXY_TIMEOUT_SECONDS: int = int(os.getenv("PROXY_TIMEOUT_SECONDS", "120"))
STREAM_EDIT_INTERVAL: float = float(os.getenv("STREAM_EDIT_INTERVAL", "1.3"))

# Ambil Multi-API Key Gemini dari .env (pisahkan dengan koma)
_raw_keys = os.getenv("GEMINI_API_KEYS", "") or os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEYS: List[str] = [k.strip() for k in _raw_keys.split(",") if k.strip()]

# Opsional: Batasi perintah ubah proxy global (/setproxy, /clearproxy) ke admin tertentu
_raw_admins = os.getenv("ADMIN_USER_IDS", "")
ADMIN_USER_IDS: set[int] = {
    int(uid.strip()) for uid in _raw_admins.split(",") if uid.strip().isdigit()
}

# Model default paling hemat token, cepat, & efisien
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

# Daftar slug model resmi yang dapat dipilih secara dinamis
AVAILABLE_GEMINI_MODELS: List[str] = [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-4-hydrogen",
]

# Kunci konfigurasi global di tabel bot_config
CONFIG_KEY_PROXY_URL = "kobold_proxy_url"

# Cache instance genai.Client per API key untuk efisiensi koneksi
_gemini_clients: Dict[str, genai.Client] = {}


# ==============================================================================
# 2. LAPISAN DATABASE SQLITE (ARSITEKTUR 3 TABEL PERSISTEN)
# ==============================================================================
class DatabaseManager:
    """
    Mengelola SQLite dengan 3 tabel wajib:
    1. chat_history  : Riwayat percakapan per user_id.
    2. bot_config    : Konfigurasi global persisten (kobold_proxy_url).
    3. user_settings : Preferensi model Gemini spesifik per user_id.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            # Mode WAL meningkatkan keamanan akses konkuren di container Linux
            conn.execute("PRAGMA journal_mode=WAL;")

            # Tabel 1: chat_history (id, user_id, role, content)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_history_user_id
                ON chat_history (user_id, id)
                """
            )

            # Tabel 2: bot_config (key TEXT PRIMARY KEY, value TEXT)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_config (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

            # Tabel 3: user_settings (preferensi model Gemini tiap user)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    gemini_model TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        logger.info("Database SQLite 3-tabel siap digunakan di: %s", self.db_path)

    # --- Operasi bot_config (Global Config) ---
    def get_config(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM bot_config WHERE key = ?", (key,)
            ).fetchone()
            if row is None or row["value"] is None or row["value"].strip() == "":
                return default
            return row["value"].strip()

    def set_config(self, key: str, value: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO bot_config (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )
            conn.commit()

    def delete_config(self, key: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM bot_config WHERE key = ?", (key,))
            conn.commit()

    # --- Operasi user_settings (Preferensi Model Tiap User) ---
    def get_user_model(self, user_id: int) -> str:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT gemini_model FROM user_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row and row["gemini_model"] in AVAILABLE_GEMINI_MODELS:
                return row["gemini_model"]
            return DEFAULT_GEMINI_MODEL

    def set_user_model(self, user_id: int, model_slug: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_settings (user_id, gemini_model, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    gemini_model = excluded.gemini_model,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, model_slug),
            )
            conn.commit()

    # --- Operasi chat_history (Memori Percakapan) ---
    def get_chat_history(
        self, user_id: int, limit: int = MAX_HISTORY_MESSAGES
    ) -> List[Tuple[str, str]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_history
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [(row["role"], row["content"]) for row in reversed(rows)]

    def add_chat_turn(self, user_id: int, user_text: str, assistant_text: str) -> None:
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)",
                [
                    (user_id, "user", user_text),
                    (user_id, "assistant", assistant_text),
                ],
            )
            conn.commit()

    def clear_chat_history(self, user_id: int) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM chat_history WHERE user_id = ?", (user_id,)
            )
            conn.commit()
            return cursor.rowcount


db = DatabaseManager(DB_PATH)


# ==============================================================================
# 3. UTILITAS KEAMANAN, URL NORMALIZER & BUILDER INLINE KEYBOARD
# ==============================================================================
def is_authorized_admin(user_id: int) -> bool:
    if not ADMIN_USER_IDS:
        return True
    return user_id in ADMIN_USER_IDS


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 10:
        return "***"
    return f"{api_key[:6]}...{api_key[-4:]}"


def normalize_proxy_url(raw_url: str) -> str:
    """Merapikan URL Cloudflare Tunnel agar selalu berakhiran '/v1/chat/completions'."""
    cleaned = raw_url.strip().rstrip("/")
    parsed = urlparse(cleaned)

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            "URL tidak valid. Awali dengan http:// atau https:// "
            "(contoh: https://tunnel-anda.trycloudflare.com)."
        )

    if cleaned.endswith("/v1/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def build_start_keyboard(has_proxy: bool) -> InlineKeyboardMarkup:
    """Membuat Inline Keyboard cepat untuk pesan /start."""
    proxy_btn = (
        InlineKeyboardButton("🔄 Kembali ke Gemini", callback_data="action:clearproxy")
        if has_proxy
        else InlineKeyboardButton("🔌 Cara Set Proxy", callback_data="menu:proxy_help")
    )
    keyboard = [
        [
            InlineKeyboardButton("🤖 Pilih Model Gemini", callback_data="menu:models"),
            proxy_btn,
        ],
        [
            InlineKeyboardButton("🧹 Reset Memori Chat", callback_data="action:reset"),
            InlineKeyboardButton("🔄 Refresh Status", callback_data="menu:start"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def build_models_keyboard(active_model: str) -> InlineKeyboardMarkup:
    """Membuat grid tombol (2 kolom per baris) untuk seluruh slug model Gemini."""
    buttons: List[InlineKeyboardButton] = []
    for slug in AVAILABLE_GEMINI_MODELS:
        label = f"✅ {slug}" if slug == active_model else slug
        buttons.append(
            InlineKeyboardButton(label, callback_data=f"setmodel:{slug}")
        )

    # Susun menjadi grid 2 tombol per baris agar rapi di layar mobile/Telegram
    rows: List[List[InlineKeyboardButton]] = [
        buttons[i : i + 2] for i in range(0, len(buttons), 2)
    ]
    rows.append([InlineKeyboardButton("⬅️ Kembali ke Menu", callback_data="menu:start")])
    return InlineKeyboardMarkup(rows)


def format_start_text(user_first_name: str, user_id: int) -> Tuple[str, bool]:
    """Menyusun teks status backend & model aktif user untuk /start."""
    name = html.escape(user_first_name or "Pengguna")
    proxy_url = db.get_config(CONFIG_KEY_PROXY_URL)
    user_model = db.get_user_model(user_id)

    if proxy_url:
        backend_status = (
            f"🔌 <b>KoboldCPP Proxy (Aktif Global)</b>\n"
            f"└ Endpoint: <code>{html.escape(proxy_url)}</code>"
        )
    else:
        backend_status = (
            f"✨ <b>Google AI Studio / Gemini (Aktif)</b>\n"
            f"└ Model Anda: <code>{html.escape(user_model)}</code>\n"
            f"└ Multi-Key Pool: <code>{len(GEMINI_API_KEYS)} API Key</code>"
        )

    text = (
        f"Halo, <b>{name}</b>! 👋\n"
        f"Selamat datang di <b>Nera, Bot AI Dual-Backend</b>.\n\n"
        f"📡 <b>Status Sistem Saat Ini:</b>\n"
        f"{backend_status}\n\n"
        f"💡 <b>Fitur Tersedia:</b>\n"
        f"• Kirim teks untuk ngobrol secara <i>real-time streaming</i>.\n"
        f"• Kirim <b>foto/gambar</b> (dengan atau tanpa caption) untuk analisis visual via Gemini.\n"
        f"• Gunakan tombol di bawah atau perintah:\n"
        f"  /models • /setmodel • /setproxy • /clearproxy • /reset"
    )
    return text, bool(proxy_url)


# ==============================================================================
# 4. STREAMING GENERATOR: GEMINI (MULTIMODAL + FAILOVER) & KOBOLDCPP PROXY
# ==============================================================================
def _get_gemini_client(api_key: str) -> genai.Client:
    """
    Menggunakan api_version='v1beta' agar alias 'gemini-flash-latest' dan
    'gemini-flash-lite-latest' tetap didukung penuh pada SDK google-genai terbaru.
    """
    if api_key not in _gemini_clients:
        _gemini_clients[api_key] = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
    return _gemini_clients[api_key]


def _should_failover_key(exc: Exception) -> bool:
    """Menentukan apakah error layak memicu perpindahan ke API key cadangan."""
    code = getattr(exc, "code", None)
    if code in (401, 403, 429, 500, 503):
        return True
    msg = str(exc).upper()
    return any(
        token in msg
        for token in ("429", "RESOURCE_EXHAUSTED", "RATE LIMIT", "QUOTA", "API_KEY_INVALID")
    )


async def stream_gemini_with_failover(
    model_slug: str,
    history: List[Tuple[str, str]],
    user_text: str,
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> AsyncGenerator[str, None]:
    """
    Streaming respons dari Google AI Studio (Gemini) dengan dukungan:
    - Riwayat chat dari SQLite
    - Input Multimodal (gambar + teks via types.Part.from_bytes)
    - Multi-API Key Failover otomatis
    """
    if not GEMINI_API_KEYS:
        raise RuntimeError(
            "GEMINI_API_KEYS belum diatur di file .env. Silakan tambahkan minimal 1 API key."
        )

    # Susun riwayat percakapan sebelumnya
    contents: List[types.Content] = []
    for role, content in history:
        gemini_role = "model" if role == "assistant" else "user"
        contents.append(
            types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=content)],
            )
        )

    # Susun pesan terbaru (mendukung teks + opsional gambar)
    current_parts: List[types.Part] = []
    if image_bytes is not None:
        current_parts.append(
            types.Part.from_bytes(data=image_bytes, mime_type=image_mime)
        )
    current_parts.append(types.Part.from_text(text=user_text))

    contents.append(types.Content(role="user", parts=current_parts))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.7,
    )

    total_keys = len(GEMINI_API_KEYS)
    last_error: Optional[Exception] = None

    for idx, api_key in enumerate(GEMINI_API_KEYS, start=1):
        masked = mask_api_key(api_key)
        emitted_any_chunk = False
        try:
            logger.info(
                "Streaming Gemini (%s) via API Key #%d/%d (%s)",
                model_slug,
                idx,
                total_keys,
                masked,
            )
            client = _get_gemini_client(api_key)
            # Pada SDK google-genai, aio generate_content_stream harus di-await sebelum async for
            response_stream = await client.aio.models.generate_content_stream(
                model=model_slug,
                contents=contents,
                config=config,
            )
            async for chunk in response_stream:
                chunk_text = getattr(chunk, "text", None)
                if chunk_text:
                    emitted_any_chunk = True
                    yield chunk_text

            if not emitted_any_chunk:
                raise ValueError(
                    "Gemini tidak mengembalikan teks (kemungkinan diblokir filter keamanan)."
                )
            return

        except Exception as exc:
            last_error = exc
            # Lakukan failover hanya jika stream belum mulai mengirim teks ke user
            if not emitted_any_chunk and _should_failover_key(exc) and idx < total_keys:
                logger.warning(
                    "API Key #%d (%s) gagal (%s). Beralih ke key cadangan berikutnya...",
                    idx,
                    masked,
                    exc,
                )
                continue
            if idx < total_keys and not emitted_any_chunk and _should_failover_key(exc):
                continue
            if not _should_failover_key(exc) or idx == total_keys:
                raise

    raise RuntimeError(
        f"Seluruh ({total_keys}) API Key Gemini gagal merespons. Error terakhir: {last_error}"
    )


def _kobold_stream_worker(
    proxy_url: str,
    payload: dict,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
) -> None:
    """
    Worker thread untuk membaca SSE streaming dari KoboldCPP (/v1/chat/completions).
    Jika server proxy mengembalikan JSON non-stream, otomatis fallback ke mode JSON biasa.
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
        }
        with requests.post(
            proxy_url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=PROXY_TIMEOUT_SECONDS,
        ) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "").lower()

            if "text/event-stream" in content_type:
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(data_str)
                        choices = chunk_json.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            piece = delta.get("content") or choices[0].get("text")
                            if piece:
                                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", piece))
                    except json.JSONDecodeError:
                        continue
            else:
                # Fallback jika endpoint proxy mengembalikan JSON sekaligus (non-streaming)
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise ValueError(f"Respons KoboldCPP tidak valid: {data}")
                msg_content = (
                    (choices[0].get("message") or {}).get("content")
                    or choices[0].get("text")
                    or ""
                )
                if msg_content:
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", str(msg_content)))

        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
    except Exception as exc:
        loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))


async def stream_kobold_proxy(
    proxy_url: str,
    history: List[Tuple[str, str]],
    user_text: str,
) -> AsyncGenerator[str, None]:
    """Mengalirkan respons dari KoboldCPP Proxy secara non-blocking."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, content in history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": "koboldcpp",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": True,
    }

    logger.info("Streaming ke KoboldCPP Proxy: %s", proxy_url)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    # Jalankan I/O requests di background thread
    worker_task = asyncio.create_task(
        asyncio.to_thread(_kobold_stream_worker, proxy_url, payload, loop, queue)
    )

    try:
        while True:
            kind, value = await queue.get()
            if kind == "chunk":
                yield value
            elif kind == "done":
                break
            elif kind == "error":
                raise value
    finally:
        await worker_task


# ==============================================================================
# 5. MESIN PENGIRIM STREAMING KE TELEGRAM (ANTI-FLOODWAIT & AUTO-SPLIT)
# ==============================================================================
async def stream_response_to_telegram(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    token_stream: AsyncGenerator[str, None],
) -> str:
    """
    Membaca potongan teks dari AI dan memperbarui pesan Telegram secara bertahap.
    - Otomatis menghormati batas edit Telegram (STREAM_EDIT_INTERVAL).
    - Otomatis memecah menjadi pesan baru jika jawaban melebihi ~3900 karakter.
    """
    chat_id = update.effective_chat.id
    reply_target = update.effective_message

    full_response_parts: List[str] = []
    current_msg_buffer = ""
    active_message: Optional[Message] = None
    last_edit_time = 0.0
    last_rendered_text = ""

    # Batas aman panjang satu pesan Telegram (maksimum resmi 4096)
    MAX_TG_CHUNK = 3900

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    async def _safe_edit(msg: Message, text: str) -> None:
        nonlocal last_rendered_text
        if text == last_rendered_text or not text.strip():
            return
        try:
            await msg.edit_text(text)
            last_rendered_text = text
        except RetryAfter as exc:
            # Jika terkena rate limit edit Telegram, tunggu sebentar lalu lanjut
            await asyncio.sleep(float(exc.retry_after) + 0.2)
        except BadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                logger.debug("BadRequest saat edit streaming: %s", exc)

    async for chunk in token_stream:
        full_response_parts.append(chunk)
        current_msg_buffer += chunk

        # Jika pesan saat ini mendekati batas 4096 karakter, kunci pesan ini lalu buat pesan baru
        if len(current_msg_buffer) >= MAX_TG_CHUNK:
            overflow_idx = current_msg_buffer.rfind("\n", 0, MAX_TG_CHUNK)
            if overflow_idx <= 0:
                overflow_idx = MAX_TG_CHUNK

            finalized_part = current_msg_buffer[:overflow_idx].rstrip()
            remainder = current_msg_buffer[overflow_idx:].lstrip()

            if active_message is None:
                active_message = await reply_target.reply_text(finalized_part)
            else:
                await _safe_edit(active_message, finalized_part)

            # Mulai pesan baru untuk sisa teks
            active_message = None
            last_rendered_text = ""
            current_msg_buffer = remainder
            last_edit_time = time.monotonic()
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            continue

        now = time.monotonic()
        if active_message is None:
            # Kirim pesan pertama secepat mungkin setelah beberapa karakter terkumpul
            if len(current_msg_buffer.strip()) >= 8:
                active_message = await reply_target.reply_text(current_msg_buffer + " ▌")
                last_rendered_text = current_msg_buffer + " ▌"
                last_edit_time = now
        elif (now - last_edit_time) >= STREAM_EDIT_INTERVAL:
            await _safe_edit(active_message, current_msg_buffer + " ▌")
            last_edit_time = now

    final_text = "".join(full_response_parts).strip()
    if not final_text:
        raise ValueError("AI menghasilkan respons kosong.")

    # Finalisasi pesan terakhir (hapus kursor '▌')
    if current_msg_buffer.strip():
        if active_message is None:
            await reply_target.reply_text(current_msg_buffer.strip())
        else:
            await _safe_edit(active_message, current_msg_buffer.strip())

    return final_text


# ==============================================================================
# 6. HANDLER PERINTAH TELEGRAM, CALLBACK QUERY & MULTIMODAL
# ==============================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /start dengan Inline Keyboard interaktif."""
    user = update.effective_user
    text, has_proxy = format_start_text(user.first_name, user.id)
    await update.effective_message.reply_text(
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=build_start_keyboard(has_proxy),
    )


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /models menampilkan grid Inline Keyboard daftar model Gemini."""
    user_id = update.effective_user.id
    active_model = db.get_user_model(user_id)
    proxy_url = db.get_config(CONFIG_KEY_PROXY_URL)

    text = (
        f"🤖 <b>Pilih Model Gemini</b>\n"
        f"Model aktif Anda saat ini: <code>{html.escape(active_model)}</code>\n\n"
        f"Klik salah satu tombol di bawah untuk mengganti model secara instan:"
    )
    if proxy_url:
        text += (
            "\n\n⚠️ <i>Info: KoboldCPP Proxy sedang aktif global. "
            "Matikan dengan /clearproxy jika ingin menggunakan Gemini.</i>"
        )

    await update.effective_message.reply_text(
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=build_models_keyboard(active_model),
    )


async def setmodel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /setmodel <slug>: mengganti model Gemini aktif untuk user tersebut."""
    user_id = update.effective_user.id
    if not context.args:
        await update.effective_message.reply_text(
            "⚠️ Format: <code>/setmodel &lt;slug&gt;</code>\n"
            "Atau ketik /models untuk memilih lewat tombol.",
            parse_mode=ParseMode.HTML,
        )
        return

    target_slug = context.args[0].strip()
    if target_slug not in AVAILABLE_GEMINI_MODELS:
        valid_list = "\n".join(
            f"• <code>{html.escape(m)}</code>" for m in AVAILABLE_GEMINI_MODELS
        )
        await update.effective_message.reply_text(
            f"❌ Model <code>{html.escape(target_slug)}</code> tidak tersedia.\n\n"
            f"Pilih salah satu:\n{valid_list}",
            parse_mode=ParseMode.HTML,
        )
        return

    db.set_user_model(user_id, target_slug)
    await update.effective_message.reply_text(
        f"✅ Model Gemini Anda berhasil diganti menjadi:\n"
        f"<code>{html.escape(target_slug)}</code>",
        parse_mode=ParseMode.HTML,
    )


async def setproxy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /setproxy <url>: menyimpan URL KoboldCPP Tunnel secara persisten ke SQLite."""
    user_id = update.effective_user.id
    if not is_authorized_admin(user_id):
        await update.effective_message.reply_text(
            "⛔ Perintah ini hanya dapat dijalankan oleh Admin yang terdaftar di .env."
        )
        return

    if not context.args:
        await update.effective_message.reply_text(
            "⚠️ Format: <code>/setproxy https://xxx.trycloudflare.com</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        normalized_url = normalize_proxy_url(context.args[0])
    except ValueError as exc:
        await update.effective_message.reply_text(f"❌ {html.escape(str(exc))}")
        return

    db.set_config(CONFIG_KEY_PROXY_URL, normalized_url)
    await update.effective_message.reply_text(
        f"✅ <b>KoboldCPP Proxy Disimpan Permanen!</b>\n"
        f"Endpoint aktif: <code>{html.escape(normalized_url)}</code>",
        parse_mode=ParseMode.HTML,
    )


async def clearproxy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /clearproxy: menghapus proxy dari bot_config & kembali ke Gemini."""
    user_id = update.effective_user.id
    if not is_authorized_admin(user_id):
        await update.effective_message.reply_text(
            "⛔ Perintah ini hanya dapat dijalankan oleh Admin yang terdaftar di .env."
        )
        return

    db.delete_config(CONFIG_KEY_PROXY_URL)
    user_model = db.get_user_model(user_id)
    await update.effective_message.reply_text(
        f"✅ <b>Proxy Dihapus!</b>\n"
        f"Bot kembali ke backend <b>Google AI Studio</b> "
        f"(Model aktif Anda: <code>{html.escape(user_model)}</code>).",
        parse_mode=ParseMode.HTML,
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /reset: menghapus riwayat chat SQLite milik user pengirim."""
    user_id = update.effective_user.id
    deleted = db.clear_chat_history(user_id)
    await update.effective_message.reply_text(
        f"🧹 Memori percakapan Anda berhasil direset ({deleted} pesan dihapus)."
    )


async def callback_query_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Menangani seluruh interaksi klik tombol Inline Keyboard dengan mulus."""
    query = update.callback_query
    if not query or not query.data:
        return

    user = query.from_user
    user_id = user.id
    data = query.data

    # Wajib menjawab callback_query agar animasi loading di tombol Telegram berhenti
    await query.answer()

    try:
        if data == "menu:start":
            text, has_proxy = format_start_text(user.first_name, user_id)
            await query.edit_message_text(
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=build_start_keyboard(has_proxy),
            )

        elif data == "menu:models":
            active_model = db.get_user_model(user_id)
            proxy_url = db.get_config(CONFIG_KEY_PROXY_URL)
            text = (
                f"🤖 <b>Pilih Model Gemini</b>\n"
                f"Model aktif Anda saat ini: <code>{html.escape(active_model)}</code>\n\n"
                f"Klik tombol di bawah untuk mengganti model:"
            )
            if proxy_url:
                text += "\n\n⚠️ <i>Catatan: Proxy KoboldCPP sedang aktif.</i>"
            await query.edit_message_text(
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=build_models_keyboard(active_model),
            )

        elif data.startswith("setmodel:"):
            target_slug = data.split(":", 1)[1]
            if target_slug not in AVAILABLE_GEMINI_MODELS:
                await query.answer("❌ Model tidak valid.", show_alert=True)
                return

            db.set_user_model(user_id, target_slug)
            text = (
                f"🤖 <b>Model Gemini Diperbarui!</b>\n"
                f"Model aktif Anda sekarang: <code>{html.escape(target_slug)}</code>"
            )
            await query.edit_message_text(
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=build_models_keyboard(target_slug),
            )

        elif data == "menu:proxy_help":
            text = (
                "🔌 <b>Panduan Pengaturan Proxy KoboldCPP</b>\n\n"
                "Untuk menyambungkan bot ke KoboldCPP lokal melalui Cloudflare Tunnel, "
                "kirim perintah dengan format:\n"
                "<code>/setproxy https://nama-tunnel.trycloudflare.com</code>\n\n"
                "URL akan otomatis dirapikan berakhiran <code>/v1/chat/completions</code> "
                "dan disimpan permanen di SQLite."
            )
            back_markup = InlineKeyboardMarkup(
                [[InlineKeyboardButton("⬅️ Kembali ke Menu", callback_data="menu:start")]]
            )
            await query.edit_message_text(
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=back_markup,
            )

        elif data == "action:clearproxy":
            if not is_authorized_admin(user_id):
                await query.answer(
                    "⛔ Hanya Admin yang dapat menghapus proxy.", show_alert=True
                )
                return
            db.delete_config(CONFIG_KEY_PROXY_URL)
            text, has_proxy = format_start_text(user.first_name, user_id)
            text = "✅ <i>Proxy dihapus. Kembali ke Google AI Studio.</i>\n\n" + text
            await query.edit_message_text(
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=build_start_keyboard(has_proxy),
            )

        elif data == "action:reset":
            deleted = db.clear_chat_history(user_id)
            text, has_proxy = format_start_text(user.first_name, user_id)
            text = (
                f"🧹 <i>Memori chat Anda direset ({deleted} pesan dihapus).</i>\n\n"
                + text
            )
            await query.edit_message_text(
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=build_start_keyboard(has_proxy),
            )

    except BadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            logger.warning("BadRequest pada CallbackQuery: %s", exc)


async def handle_text_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Menangani pesan teks biasa dan merutekannya ke KoboldCPP atau Gemini (Streaming)."""
    if not update.effective_message or not update.effective_message.text:
        return

    user_id = update.effective_user.id
    user_text = update.effective_message.text.strip()
    if not user_text:
        return

    proxy_url = db.get_config(CONFIG_KEY_PROXY_URL)
    history = db.get_chat_history(user_id=user_id, limit=MAX_HISTORY_MESSAGES)

    try:
        if proxy_url and proxy_url.endswith("/v1/chat/completions"):
            token_stream = stream_kobold_proxy(proxy_url, history, user_text)
        else:
            user_model = db.get_user_model(user_id)
            token_stream = stream_gemini_with_failover(
                model_slug=user_model,
                history=history,
                user_text=user_text,
            )

        final_reply = await stream_response_to_telegram(update, context, token_stream)
        db.add_chat_turn(user_id=user_id, user_text=user_text, assistant_text=final_reply)

    except requests.exceptions.Timeout:
        logger.error("Timeout menghubungi KoboldCPP Proxy.")
        await update.effective_message.reply_text(
            "⏱ Koneksi ke KoboldCPP Proxy timeout. Pastikan PC lokal & tunnel aktif."
        )
    except requests.exceptions.RequestException as exc:
        logger.error("Error HTTP KoboldCPP Proxy: %s", exc)
        await update.effective_message.reply_text(
            "⚠️ Gagal terhubung ke KoboldCPP Proxy. Gunakan /clearproxy untuk kembali ke Gemini."
        )
    except genai_errors.APIError as exc:
        logger.error("Gemini APIError: %s", exc)
        await update.effective_message.reply_text(
            f"⚠️ Gemini API error (Kode {getattr(exc, 'code', 'Unknown')}). "
            "Coba ganti model lewat /models."
        )
    except Exception as exc:
        logger.exception("Error memproses pesan teks user %s: %s", user_id, exc)
        await update.effective_message.reply_text(f"❌ Terjadi kesalahan: {exc}")


async def handle_photo_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Menangani input foto (filters.PHOTO):
    - Mengunduh gambar ke file sementara (auto-delete).
    - Mengonversi menjadi types.Part.from_bytes untuk SDK google-genai.
    - Mengirim bersama caption (atau prompt default) ke Gemini secara streaming.
    """
    message = update.effective_message
    if not message or not message.photo:
        return

    user_id = update.effective_user.id
    caption = (message.caption or "").strip()
    prompt_text = caption if caption else "Analisis dan jelaskan gambar ini secara detail."

    # Cek jika KoboldCPP Proxy sedang aktif (analisis visual membutuhkan Gemini)
    proxy_url = db.get_config(CONFIG_KEY_PROXY_URL)
    if proxy_url:
        await message.reply_text(
            "ℹ️ Analisis gambar multimodal menggunakan <b>Google Gemini Vision</b>.\n"
            "Saat ini <b>KoboldCPP Proxy</b> sedang aktif. Jalankan /clearproxy terlebih dahulu "
            "untuk menganalisis gambar.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        # Ambil resolusi foto tertinggi (elemen terakhir pada array photo)
        highest_res_photo = message.photo[-1]
        tg_file = await highest_res_photo.get_file()

        # Unduh sementara ke disk lalu baca bytes, otomatis terhapus saat blok 'with' selesai
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp_file:
            await tg_file.download_to_drive(custom_path=tmp_file.name)
            image_bytes = Path(tmp_file.name).read_bytes()

        history = db.get_chat_history(user_id=user_id, limit=MAX_HISTORY_MESSAGES)
        user_model = db.get_user_model(user_id)

        token_stream = stream_gemini_with_failover(
            model_slug=user_model,
            history=history,
            user_text=prompt_text,
            image_bytes=image_bytes,
            image_mime="image/jpeg",
        )

        final_reply = await stream_response_to_telegram(update, context, token_stream)

        # Catat ke SQLite dengan penanda bahwa user mengirim gambar
        saved_user_msg = f"[Mengirim Foto] {prompt_text}"
        db.add_chat_turn(
            user_id=user_id,
            user_text=saved_user_msg,
            assistant_text=final_reply,
        )

    except genai_errors.APIError as exc:
        logger.error("Gemini APIError pada gambar: %s", exc)
        await message.reply_text(
            f"⚠️ Gagal menganalisis gambar (Kode {getattr(exc, 'code', 'Unknown')}). "
            "Pastikan model mendukung input gambar atau coba model lain via /models."
        )
    except Exception as exc:
        logger.exception("Error memproses foto user %s: %s", user_id, exc)
        await message.reply_text(f"❌ Gagal memproses gambar: {exc}")


async def global_error_handler(
    update: object, context: ContextTypes.DEFAULT_TYPE
) -> None:
    logger.exception("Unhandled Telegram Bot error: %s", context.error)


async def post_init_setup(application: Application) -> None:
    """Otomatis mendaftarkan menu slash command ke Telegram API saat bot start."""
    commands = [
        BotCommand("start", "Menu utama, status backend & tombol cepat"),
        BotCommand("models", "Pilih model Gemini lewat tombol interaktif"),
        BotCommand("setmodel", "Ganti model Gemini secara manual (/setmodel <slug>)"),
        BotCommand("setproxy", "Set URL Cloudflare KoboldCPP (/setproxy <url>)"),
        BotCommand("clearproxy", "Hapus proxy & kembali ke Google AI Studio"),
        BotCommand("reset", "Hapus riwayat memori chat Anda"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Menu perintah Telegram berhasil disinkronkan.")


# ==============================================================================
# 7. ENTRYPOINT APLIKASI
# ==============================================================================
def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            f"TELEGRAM_BOT_TOKEN tidak ditemukan di {ENV_PATH}. Harap lengkapi file .env."
        )

    if not GEMINI_API_KEYS:
        logger.warning(
            "GEMINI_API_KEYS kosong di .env! Mode Google AI Studio tidak akan berjalan sebelum diisi."
        )

    application: Application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init_setup)
        .build()
    )

    # Command Handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("models", models_command))
    application.add_handler(CommandHandler("setmodel", setmodel_command))
    application.add_handler(CommandHandler("setproxy", setproxy_command))
    application.add_handler(CommandHandler("clearproxy", clearproxy_command))
    application.add_handler(CommandHandler("reset", reset_command))

    # Inline Keyboard Callback Handler
    application.add_handler(CallbackQueryHandler(callback_query_handler))

    # Multimodal Photo Handler & Text Message Handler
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message)
    )

    application.add_error_handler(global_error_handler)

    logger.info("Bot Telegram Dual-Backend aktif (Polling) | Base Dir: %s", BASE_DIR)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
