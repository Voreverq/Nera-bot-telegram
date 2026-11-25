import asyncio
import base64
import contextlib
import html
import json
import logging
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, NamedTuple, Optional, Tuple
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
# 1. KONFIGURASI PATH RELATIF & ENVIRONMENT
# ==============================================================================
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

MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
PROXY_TIMEOUT_SECONDS: int = int(os.getenv("PROXY_TIMEOUT_SECONDS", "120"))
STREAM_EDIT_INTERVAL: float = float(os.getenv("STREAM_EDIT_INTERVAL", "1.3"))

# Multi-API Key Gemini dari .env (dipisah koma)
_raw_keys = os.getenv("GEMINI_API_KEYS", "") or os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEYS: List[str] = [k.strip() for k in _raw_keys.split(",") if k.strip()]

# Opsional: Batasi perintah ubah proxy/style global ke User ID Admin tertentu
_raw_admins = os.getenv("ADMIN_USER_IDS", "")
ADMIN_USER_IDS: set[int] = {
    int(uid.strip()) for uid in _raw_admins.split(",") if uid.strip().isdigit()
}

# ==============================================================================
# 2. DEFAULT HARDCODED SYSTEM PROMPT, PERSONALITY & MODEL CONFIG
# ==============================================================================
# Nilai default hardcoded (dapat di-overwrite lewat /setprompt & /setpersona)
DEFAULT_SYSTEM_PROMPT: str = os.getenv(
    "SYSTEM_PROMPT",
    (
        "Kamu adalah asisten AI dual-backend yang cerdas, teliti, dan solutif. "
        "Berikan jawaban yang terstruktur untuk topik teknis/coding, analisis file "
        "secara akurat, dan katakan jujur jika suatu informasi tidak tersedia."
    ),
).strip()

DEFAULT_PERSONALITY: str = os.getenv(
    "DEFAULT_PERSONALITY",
    (
        "Gunakan bahasa Indonesia yang natural, ramah, jelas, tidak kaku, "
        "serta ringkas untuk pertanyaan sederhana dan mendalam untuk topik teknis."
    ),
).strip()

# Konfigurasi Smart Router & Daftar Model
SMART_ROUTER_MODE = "auto"
DECISION_ROUTER_MODEL = "gemini-decision-1"
HEAVY_GEMINI_MODEL = "gemini-4-hydrogen"
LITE_GEMINI_MODEL = "gemini-3.5-flash-lite"

# Default user sekarang menggunakan 'auto' (Smart Router gemini-decision-1)
DEFAULT_USER_MODEL_MODE = SMART_ROUTER_MODE

AVAILABLE_GEMINI_MODELS: List[str] = [
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-4-hydrogen",
]
VALID_MODEL_MODES: List[str] = [SMART_ROUTER_MODE] + AVAILABLE_GEMINI_MODELS

# Level berpikir (thinking) untuk model Gemini 4 seperti gemini-4-hydrogen.
# 'auto' = router (gemini-decision-1) menilai kedalaman penalaran tiap pesan.
THINKING_MODE_AUTO = "auto"
THINKING_LEVELS: List[str] = ["low", "medium", "high", "xhigh"]
VALID_THINKING_MODES: List[str] = [THINKING_MODE_AUTO] + THINKING_LEVELS
DEFAULT_THINKING_MODE = THINKING_MODE_AUTO
# Dipakai jika router tidak memberi saran (mis. mode model manual + thinking 'auto')
FALLBACK_THINKING_LEVEL = "medium"
# Router sengaja tidak menyarankan xhigh (terlalu lambat untuk chat Telegram)
ROUTER_THINKING_LEVELS: List[str] = ["low", "medium", "high"]
THINKING_CAPABLE_PREFIXES: Tuple[str, ...] = ("gemini-4-",)
THINKING_LABELS: Dict[str, str] = {
    THINKING_MODE_AUTO: "Auto (router)",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "XHigh",
}

# Batas input untuk Smart Router agar tidak melewati jendela konteks/ukuran request
ROUTER_MAX_TEXT_CHARS = 4000
ROUTER_MAX_ATTACHMENT_BYTES = 4 * 1024 * 1024
# Batas file teks/kode (dikirim utuh sebagai teks ke model)
MAX_TEXT_FILE_BYTES = 1 * 1024 * 1024

# Kunci konfigurasi persisten di tabel bot_config
CONFIG_KEY_PROXY_URL = "kobold_proxy_url"
CONFIG_KEY_SYSTEM_PROMPT = "custom_system_prompt"
CONFIG_KEY_PERSONALITY = "custom_personality"

# Cache instance genai.Client per API key
_gemini_clients: Dict[str, genai.Client] = {}


# ==============================================================================
# 3. LAPISAN DATABASE SQLITE (3 TABEL PERSISTEN)
# ==============================================================================
class DatabaseManager:
    """Mengelola SQLite: chat_history, bot_config, dan user_settings."""

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
            conn.execute("PRAGMA journal_mode=WAL;")

            # Tabel 1: chat_history
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

            # Tabel 2: bot_config (proxy, custom system prompt, custom personality)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_config (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

            # Tabel 3: user_settings (mode 'auto' atau slug model Gemini tiap user)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    gemini_model TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Migrasi ringan: kolom thinking_level untuk database lama
            existing_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(user_settings)").fetchall()
            }
            if "thinking_level" not in existing_cols:
                conn.execute("ALTER TABLE user_settings ADD COLUMN thinking_level TEXT")
            conn.commit()
        logger.info("Database SQLite siap di: %s", self.db_path)

    # --- bot_config ---
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

    # --- user_settings ---
    def get_user_model_mode(self, user_id: int) -> str:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT gemini_model FROM user_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row and row["gemini_model"] in VALID_MODEL_MODES:
                return row["gemini_model"]
            return DEFAULT_USER_MODEL_MODE

    def set_user_model_mode(self, user_id: int, mode_or_slug: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_settings (user_id, gemini_model, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    gemini_model = excluded.gemini_model,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, mode_or_slug),
            )
            conn.commit()

    def get_user_thinking_mode(self, user_id: int) -> str:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT thinking_level FROM user_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row and row["thinking_level"] in VALID_THINKING_MODES:
                return row["thinking_level"]
            return DEFAULT_THINKING_MODE

    def set_user_thinking_mode(self, user_id: int, mode: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_settings (user_id, gemini_model, thinking_level, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    thinking_level = excluded.thinking_level,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, DEFAULT_USER_MODEL_MODE, mode),
            )
            conn.commit()

    # --- chat_history ---
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
# 4. RAKITAN PROMPT DINAMIS, SAFETY SETTINGS & UTILITAS
# ==============================================================================
def is_authorized_admin(user_id: int) -> bool:
    if not ADMIN_USER_IDS:
        return True
    return user_id in ADMIN_USER_IDS


def model_supports_thinking_level(model_slug: str) -> bool:
    """True jika model mendukung thinking_level (keluarga Gemini 4)."""
    return model_slug.startswith(THINKING_CAPABLE_PREFIXES)


def resolve_thinking_level(model_slug: str, preferred: Optional[str]) -> Optional[str]:
    """Level final untuk dikirim ke API, atau None jika model tidak mendukungnya."""
    if not model_supports_thinking_level(model_slug):
        return None
    if preferred in THINKING_LEVELS:
        return preferred
    return FALLBACK_THINKING_LEVEL


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 10:
        return "***"
    return f"{api_key[:6]}...{api_key[-4:]}"


def get_effective_prompts() -> Tuple[str, str, bool, bool]:
    """
    Mengambil System Prompt & Personality yang aktif.
    Jika ada di SQLite bot_config, otomatis menimpa (overwrite) nilai hardcoded.
    Return: (effective_prompt, effective_personality, is_custom_prompt, is_custom_persona)
    """
    custom_prompt = db.get_config(CONFIG_KEY_SYSTEM_PROMPT)
    custom_persona = db.get_config(CONFIG_KEY_PERSONALITY)

    effective_prompt = (
        custom_prompt if custom_prompt is not None else DEFAULT_SYSTEM_PROMPT
    )
    effective_persona = (
        custom_persona if custom_persona is not None else DEFAULT_PERSONALITY
    )
    return (
        effective_prompt,
        effective_persona,
        custom_prompt is not None,
        custom_persona is not None,
    )


def build_combined_system_instruction() -> str:
    """Merakit gabungan System Prompt + Personality untuk dikirim ke Gemini & KoboldCPP."""
    sys_prompt, personality, _, _ = get_effective_prompts()
    return (
        f"[SYSTEM RULES & INSTRUCTIONS]\n{sys_prompt}\n\n"
        f"[PERSONALITY & TONE]\n{personality}"
    )


def get_gemini_safety_settings() -> List[types.SafetySetting]:
    """
    Konfigurasi SafetySetting Gemini API ke BLOCK_NONE pada seluruh kategori
    yang dapat dikonfigurasi.
    Catatan: Filter inti/hard safety tingkat platform Google tetap berlaku.
    Untuk obrolan santai/roleplay yang lebih bebas, gunakan backend KoboldCPP.
    """
    return [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]


def normalize_proxy_url(raw_url: str) -> str:
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
    proxy_btn = (
        InlineKeyboardButton("🔄 Kembali ke Gemini", callback_data="action:clearproxy")
        if has_proxy
        else InlineKeyboardButton("🔌 Cara Set Proxy", callback_data="menu:proxy_help")
    )
    keyboard = [
        [
            InlineKeyboardButton("🤖 Model & Smart Router", callback_data="menu:models"),
            proxy_btn,
        ],
        [
            InlineKeyboardButton("🎭 Prompt & Persona", callback_data="menu:style"),
            InlineKeyboardButton("🧹 Reset Memori", callback_data="action:reset"),
        ],
        [
            InlineKeyboardButton("🧩 Level Berpikir", callback_data="menu:thinking"),
            InlineKeyboardButton("🔄 Refresh Status", callback_data="menu:start"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def build_models_keyboard(active_mode: str) -> InlineKeyboardMarkup:
    """Tombol full-width untuk Auto Smart Router + grid 2 kolom untuk model manual."""
    auto_label = (
        f"✅ 🧠 Auto Smart Router ({DECISION_ROUTER_MODEL})"
        if active_mode == SMART_ROUTER_MODE
        else f"🧠 Auto Smart Router ({DECISION_ROUTER_MODEL})"
    )
    rows: List[List[InlineKeyboardButton]] = [
        [InlineKeyboardButton(auto_label, callback_data=f"setmodel:{SMART_ROUTER_MODE}")]
    ]

    buttons: List[InlineKeyboardButton] = []
    for slug in AVAILABLE_GEMINI_MODELS:
        label = f"✅ {slug}" if slug == active_mode else slug
        buttons.append(InlineKeyboardButton(label, callback_data=f"setmodel:{slug}"))

    rows.extend(buttons[i : i + 2] for i in range(0, len(buttons), 2))
    rows.append([InlineKeyboardButton("🧩 Level Berpikir", callback_data="menu:thinking")])
    rows.append([InlineKeyboardButton("⬅️ Kembali ke Menu", callback_data="menu:start")])
    return InlineKeyboardMarkup(rows)


def build_thinking_keyboard(active_mode: str) -> InlineKeyboardMarkup:
    """Tombol pilihan level thinking: Auto (router) + Low/Medium/High/XHigh."""

    def _btn(mode: str, label: str) -> InlineKeyboardButton:
        text = f"✅ {label}" if mode == active_mode else label
        return InlineKeyboardButton(text, callback_data=f"setthink:{mode}")

    rows: List[List[InlineKeyboardButton]] = [
        [_btn(THINKING_MODE_AUTO, "🧠 Auto (dipilih router)")],
        [_btn("low", "⚡ Low"), _btn("medium", "⚖️ Medium")],
        [_btn("high", "🧩 High"), _btn("xhigh", "🔬 XHigh")],
        [InlineKeyboardButton("⬅️ Kembali ke Menu", callback_data="menu:start")],
    ]
    return InlineKeyboardMarkup(rows)


def format_thinking_text(active_mode: str) -> str:
    label = html.escape(THINKING_LABELS.get(active_mode, active_mode))
    return (
        "🧩 <b>Level Berpikir (Thinking)</b>\n"
        f"Aktif: <code>{label}</code>\n\n"
        "• <b>Auto</b>: router menilai kedalaman penalaran (low–high) tiap pesan.\n"
        "• <b>Low</b>: tercepat • <b>Medium</b>: seimbang • <b>High</b>: lebih teliti • "
        "<b>XHigh</b>: paling dalam, tapi paling lambat.\n\n"
        "<i>Hanya berlaku untuk model Gemini 4 (mis. gemini-4-hydrogen); "
        "model lain mengabaikannya.</i>"
    )


def format_start_text(user_first_name: str, user_id: int) -> Tuple[str, bool]:
    name = html.escape(user_first_name or "Pengguna")
    proxy_url = db.get_config(CONFIG_KEY_PROXY_URL)
    user_mode = db.get_user_model_mode(user_id)
    thinking_label = THINKING_LABELS.get(
        db.get_user_thinking_mode(user_id), DEFAULT_THINKING_MODE
    )
    _, _, is_custom_p, is_custom_pers = get_effective_prompts()

    if proxy_url:
        backend_status = (
            f"🔌 <b>KoboldCPP Proxy (Aktif Global)</b>\n"
            f"└ Endpoint: <code>{html.escape(proxy_url)}</code>\n"
            f"└ Cocok untuk obrolan santai &amp; roleplay bebas."
        )
    elif user_mode == SMART_ROUTER_MODE:
        backend_status = (
            f"🧠 <b>Google Gemini Smart Router (Aktif)</b>\n"
            f"└ Router: <code>{DECISION_ROUTER_MODEL}</code>\n"
            f"└ HEAVY → <code>{HEAVY_GEMINI_MODEL}</code>\n"
            f"└ LITE → <code>{LITE_GEMINI_MODEL}</code>\n"
            f"└ Thinking: <code>{html.escape(thinking_label)}</code>\n"
            f"└ Key Pool: <code>{len(GEMINI_API_KEYS)} API Key</code>"
        )
    else:
        backend_status = (
            f"✨ <b>Google AI Studio / Gemini (Manual)</b>\n"
            f"└ Model Anda: <code>{html.escape(user_mode)}</code>\n"
            + (
                f"└ Thinking: <code>{html.escape(thinking_label)}</code>\n"
                if model_supports_thinking_level(user_mode)
                else ""
            )
            + f"└ Key Pool: <code>{len(GEMINI_API_KEYS)} API Key</code>"
        )

    prompt_badge = "Custom (DB)" if is_custom_p else "Default (Hardcoded)"
    persona_badge = "Custom (DB)" if is_custom_pers else "Default (Hardcoded)"

    text = (
        f"Halo, <b>{name}</b>! 👋\n"
        f"Selamat datang di <b>Nera, Bot AI Dual-Backend</b>.\n\n"
        f"📡 <b>Status Sistem Saat Ini:</b>\n"
        f"{backend_status}\n\n"
        f"🎭 <b>Instruksi Aktif:</b>\n"
        f"└ System Prompt: <code>{prompt_badge}</code>\n"
        f"└ Personality: <code>{persona_badge}</code>\n\n"
        f"💡 <b>Kirim teks, foto, atau dokumen (PDF/kode)</b> untuk langsung memulai.\n"
        f"Perintah: /models • /thinking • /style • /setprompt • /setpersona • /resetstyle • /setproxy • /clearproxy • /reset"
    )
    return text, bool(proxy_url)


# ==============================================================================
# 5. SMART ROUTER (GEMINI DECISION 1) & MULTI-KEY FAILOVER
# ==============================================================================
def _get_gemini_client(api_key: str) -> genai.Client:
    """
    Menggunakan api_version='v1beta' karena Decisions API (gemini-decision-1)
    dan alias '-latest' berada di endpoint v1beta.
    """
    if api_key not in _gemini_clients:
        _gemini_clients[api_key] = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
    return _gemini_clients[api_key]


class RouteResult(NamedTuple):
    model: str
    label: str
    thinking_level: Optional[str] = None


def _should_failover_key(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code in (401, 403, 408, 429, 500, 502, 503, 504):
        return True
    msg = str(exc).upper()
    return any(
        token in msg
        for token in ("429", "RESOURCE_EXHAUSTED", "RATE LIMIT", "QUOTA", "API_KEY_INVALID")
    )


def _clip_for_router(text: str, limit: int = ROUTER_MAX_TEXT_CHARS) -> str:
    """Potong teks panjang (mis. isi file kode) agar router tetap cepat & dalam batas konteks."""
    text = text or ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[truncated; original length {len(text)} characters]"


def _build_router_input(
    user_text: str,
    attachment_bytes: Optional[bytes],
    attachment_mime: Optional[str],
    attachment_name: Optional[str],
) -> Any:
    """Susun input Decisions API: teks saja, atau [teks, media] untuk gambar/PDF kecil."""
    clipped = _clip_for_router(user_text)
    if not (attachment_bytes and attachment_mime):
        return clipped

    label = attachment_name or "attachment"
    if attachment_mime.startswith("image/"):
        media_type: Optional[str] = "image"
    elif attachment_mime == "application/pdf":
        media_type = "document"
    else:
        media_type = None

    # Lampiran tak didukung / terlalu besar: kirim deskripsi saja (bukan byte-nya),
    # supaya request tidak gagal lalu jatuh ke fallback LITE.
    if media_type is None or len(attachment_bytes) > ROUTER_MAX_ATTACHMENT_BYTES:
        size_mb = len(attachment_bytes) / (1024 * 1024)
        return (
            f"User attached a file '{label}' ({attachment_mime}, {size_mb:.1f} MB; "
            f"content not included).\nUser message: {clipped}"
        )

    media_block: Dict[str, Any] = {
        "type": media_type,
        "data": base64.b64encode(attachment_bytes).decode("ascii"),
        "mime_type": attachment_mime,
    }
    if media_type == "image":
        # Untuk routing, resolusi rendah cukup (~4x lebih hemat token gambar)
        media_block["resolution"] = "low"

    return [
        {
            "type": "text",
            "text": f"Attachment: {label}\nUser message: {clipped}\nAttached content:\n",
        },
        media_block,
    ]


ROUTER_QUESTIONS: Dict[str, Any] = {
    "route": {
        "type": "choice",
        "instructions": (
            "Determine whether this user request requires the HEAVY reasoning model "
            "or the LITE fast assistant model."
        ),
        "criteria": {
            "HEAVY": (
                "Programming, coding, debugging, software architecture, complex math/logic, "
                "deep reasoning, multi-step analysis, tool calling, or analyzing attached PDF/code/complex files."
            ),
            "LITE": (
                "Quick general questions, greetings, casual chat, short summaries, "
                "simple translations, basic definitions, or lightweight assistant tasks."
            ),
        },
    },
    # Pertanyaan ke-2 pada panggilan yang sama (input hanya ditagih sekali).
    # Skor 0..2 dipetakan ke ROUTER_THINKING_LEVELS (low/medium/high).
    "depth": {
        "type": "score",
        "instructions": "How much deliberate reasoning effort does a good answer need?",
        "criteria": [
            "Low: simple lookup, chat, or short rewrite; little reasoning needed.",
            "Medium: moderate analysis, explanation, or a small code change.",
            "High: complex debugging, multi-step logic, architecture, or long document analysis.",
        ],
    },
}

_router_fallback_count = 0


def _thinking_level_from_depth(result: Any) -> Optional[str]:
    try:
        score = float(result.answers["depth"].score)
    except Exception:
        return None
    idx = max(0, min(len(ROUTER_THINKING_LEVELS) - 1, int(score + 0.5)))
    return ROUTER_THINKING_LEVELS[idx]


async def route_with_gemini_decision(
    user_text: str,
    attachment_bytes: Optional[bytes] = None,
    attachment_mime: Optional[str] = None,
    attachment_name: Optional[str] = None,
) -> RouteResult:
    """
    Mengevaluasi pesan masuk menggunakan `gemini-decision-1` (Decisions API):
      - HEAVY -> gemini-4-hydrogen (coding, PDF/file kompleks, penalaran dalam)
      - LITE  -> gemini-3.5-flash-lite (pertanyaan cepat, obrolan ringan)
    Sekaligus menyarankan level thinking (low/medium/high) untuk rute HEAVY.
    Fallback ke LITE jika Decisions API gagal pada semua key (dicatat sebagai ERROR).
    """
    global _router_fallback_count

    if not GEMINI_API_KEYS:
        return RouteResult(LITE_GEMINI_MODEL, "FALLBACK_NO_KEYS")

    decision_input = _build_router_input(
        user_text, attachment_bytes, attachment_mime, attachment_name
    )

    for idx, api_key in enumerate(GEMINI_API_KEYS, start=1):
        masked = mask_api_key(api_key)
        try:
            client = _get_gemini_client(api_key)
            result = await client.aio.decisions.create(
                model=DECISION_ROUTER_MODEL,
                input=decision_input,
                questions=ROUTER_QUESTIONS,
            )
            answer = result.answers["route"]
            tier = str(answer.choice).upper()  # "HEAVY" atau "LITE"
            confidence = float(getattr(answer, "confidence", 0.0) or 0.0)

            is_heavy = tier == "HEAVY"
            target_model = HEAVY_GEMINI_MODEL if is_heavy else LITE_GEMINI_MODEL
            thinking_level = _thinking_level_from_depth(result) if is_heavy else None
            logger.info(
                "Smart Router (%s | Key #%d %s): %s -> %s (confidence=%.2f, thinking=%s)",
                DECISION_ROUTER_MODEL,
                idx,
                masked,
                tier,
                target_model,
                confidence,
                thinking_level or "-",
            )
            return RouteResult(target_model, f"{tier} ({confidence:.2f})", thinking_level)

        except Exception as exc:
            if _should_failover_key(exc) and idx < len(GEMINI_API_KEYS):
                logger.warning(
                    "Decision Router gagal pada Key #%d (%s): %s. Mencoba key berikutnya...",
                    idx,
                    masked,
                    exc,
                )
                continue

            _router_fallback_count += 1
            logger.error(
                "Smart Router (%s) GAGAL (%s: %s) -> fallback %s "
                "[fallback ke-%d sejak bot start]. Jika terjadi pada semua pesan, "
                "periksa skema request Decisions API, api_version v1beta, dan versi google-genai.",
                DECISION_ROUTER_MODEL,
                type(exc).__name__,
                exc,
                LITE_GEMINI_MODEL,
                _router_fallback_count,
            )
            break

    return RouteResult(LITE_GEMINI_MODEL, "FALLBACK_LITE")


# ==============================================================================
# 6. STREAMING BACKEND: GEMINI & KOBOLDCPP PROXY
# ==============================================================================
async def stream_gemini_with_failover(
    model_slug: str,
    history: List[Tuple[str, str]],
    user_text: str,
    attachment_bytes: Optional[bytes] = None,
    attachment_mime: Optional[str] = None,
    thinking_level: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Streaming respons dari Google Gemini dengan:
    - thinking_level (low/medium/high/xhigh) khusus model Gemini 4
    - Combined System Instruction (Custom DB Overwrite atau Default Hardcoded)
    - SafetySetting BLOCK_NONE
    - Multimodal support (Gambar / PDF)
    - Multi-API Key Failover
    """
    if not GEMINI_API_KEYS:
        raise RuntimeError("GEMINI_API_KEYS belum diatur di file .env.")

    contents: List[types.Content] = []
    for role, content in history:
        gemini_role = "model" if role == "assistant" else "user"
        contents.append(
            types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=content)],
            )
        )

    current_parts: List[types.Part] = []
    if attachment_bytes is not None and attachment_mime:
        current_parts.append(
            types.Part.from_bytes(data=attachment_bytes, mime_type=attachment_mime)
        )
    current_parts.append(types.Part.from_text(text=user_text))
    contents.append(types.Content(role="user", parts=current_parts))

    config_kwargs: Dict[str, Any] = {
        "system_instruction": build_combined_system_instruction(),
        "safety_settings": get_gemini_safety_settings(),
        "temperature": 0.7,
    }
    effective_level = resolve_thinking_level(model_slug, thinking_level)
    if effective_level:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=effective_level.upper()
        )
    config = types.GenerateContentConfig(**config_kwargs)

    total_keys = len(GEMINI_API_KEYS)
    last_error: Optional[Exception] = None

    for idx, api_key in enumerate(GEMINI_API_KEYS, start=1):
        masked = mask_api_key(api_key)
        emitted_any_chunk = False
        try:
            logger.info(
                "Streaming Gemini (%s | thinking=%s) via Key #%d/%d (%s)",
                model_slug,
                effective_level or "-",
                idx,
                total_keys,
                masked,
            )
            client = _get_gemini_client(api_key)
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
                    "Gemini tidak mengembalikan teks (kemungkinan terfilter oleh core safety platform). "
                    "Untuk obrolan santai/roleplay yang lebih bebas, Anda dapat beralih ke KoboldCPP via /setproxy."
                )
            return

        except Exception as exc:
            last_error = exc
            if not emitted_any_chunk and _should_failover_key(exc) and idx < total_keys:
                logger.warning(
                    "Key #%d (%s) gagal (%s). Beralih ke key berikutnya...",
                    idx,
                    masked,
                    exc,
                )
                continue
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
    """Mengalirkan respons dari KoboldCPP Proxy dengan system prompt & personality aktif."""
    messages = [{"role": "system", "content": build_combined_system_instruction()}]
    for role, content in history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": "koboldcpp",
        "messages": messages,
        "temperature": 0.75,
        "max_tokens": 1024,
        "stream": True,
    }

    logger.info("Streaming ke KoboldCPP Proxy: %s", proxy_url)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

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
# 7. STREAMING RENDERER KE TELEGRAM (ANTI-FLOODWAIT & AUTO-SPLIT)
# ==============================================================================
async def _keep_typing(bot: Any, chat_id: int, stop: asyncio.Event) -> None:
    """Kirim indikator 'mengetik' tiap ~4 dtk selama model berpikir (indikator Telegram hilang setelah ~5 dtk)."""
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            continue


async def stream_response_to_telegram(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    token_stream: AsyncGenerator[str, None],
) -> str:
    stop = asyncio.Event()
    typing_task = asyncio.create_task(
        _keep_typing(context.bot, update.effective_chat.id, stop)
    )
    try:
        return await _render_stream_to_telegram(update, context, token_stream)
    finally:
        stop.set()
        typing_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await typing_task


async def _render_stream_to_telegram(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    token_stream: AsyncGenerator[str, None],
) -> str:
    chat_id = update.effective_chat.id
    reply_target = update.effective_message

    full_response_parts: List[str] = []
    current_msg_buffer = ""
    active_message: Optional[Message] = None
    last_edit_time = 0.0
    last_rendered_text = ""
    MAX_TG_CHUNK = 3900

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    async def _safe_edit(msg: Message, text: str) -> None:
        nonlocal last_rendered_text
        if text == last_rendered_text or not text.strip():
            return
        # Coba ulang setelah FloodWait agar edit terakhir (teks final) tidak hilang
        for _ in range(3):
            try:
                await msg.edit_text(text)
                last_rendered_text = text
                return
            except RetryAfter as exc:
                await asyncio.sleep(float(exc.retry_after) + 0.2)
            except BadRequest as exc:
                if "message is not modified" not in str(exc).lower():
                    logger.debug("BadRequest saat edit streaming: %s", exc)
                return

    async for chunk in token_stream:
        full_response_parts.append(chunk)
        current_msg_buffer += chunk

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

            active_message = None
            last_rendered_text = ""
            current_msg_buffer = remainder
            last_edit_time = time.monotonic()
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            continue

        now = time.monotonic()
        if active_message is None:
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

    if current_msg_buffer.strip():
        if active_message is None:
            await reply_target.reply_text(current_msg_buffer.strip())
        else:
            await _safe_edit(active_message, current_msg_buffer.strip())

    return final_text


# ==============================================================================
# 8. HANDLER PERINTAH TELEGRAM (TERMASUK /setprompt, /setpersona, /resetstyle)
# ==============================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    text, has_proxy = format_start_text(user.first_name, user.id)
    await update.effective_message.reply_text(
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=build_start_keyboard(has_proxy),
    )


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    active_mode = db.get_user_model_mode(user_id)
    proxy_url = db.get_config(CONFIG_KEY_PROXY_URL)

    mode_desc = (
        f"🧠 Auto Smart Router (<code>{DECISION_ROUTER_MODEL}</code>)"
        if active_mode == SMART_ROUTER_MODE
        else f"<code>{html.escape(active_mode)}</code>"
    )
    text = (
        f"🤖 <b>Pilih Mode / Model Gemini</b>\n"
        f"Mode aktif Anda: {mode_desc}\n\n"
        f"• <b>Auto Smart Router</b>: Otomatis memilih <code>{HEAVY_GEMINI_MODEL}</code> (tugas berat/coding/PDF) "
        f"atau <code>{LITE_GEMINI_MODEL}</code> (obrolan cepat).\n"
        f"• Atau pilih model spesifik di bawah:"
    )
    if proxy_url:
        text += (
            "\n\n⚠️ <i>KoboldCPP Proxy sedang aktif global. "
            "Jalankan /clearproxy untuk kembali ke Gemini.</i>"
        )

    await update.effective_message.reply_text(
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=build_models_keyboard(active_mode),
    )


async def setmodel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not context.args:
        await update.effective_message.reply_text(
            "⚠️ Format: <code>/setmodel auto</code> atau <code>/setmodel &lt;slug&gt;</code>\n"
            "Ketik /models untuk memilih via tombol.",
            parse_mode=ParseMode.HTML,
        )
        return

    target_mode = context.args[0].strip()
    if target_mode not in VALID_MODEL_MODES:
        valid_list = "\n".join(
            f"• <code>{html.escape(m)}</code>" for m in VALID_MODEL_MODES
        )
        await update.effective_message.reply_text(
            f"❌ Mode/Model <code>{html.escape(target_mode)}</code> tidak dikenali.\n\n"
            f"Pilihan valid:\n{valid_list}",
            parse_mode=ParseMode.HTML,
        )
        return

    db.set_user_model_mode(user_id, target_mode)
    await update.effective_message.reply_text(
        f"✅ Mode model Anda diubah ke: <code>{html.escape(target_mode)}</code>",
        parse_mode=ParseMode.HTML,
    )


async def thinking_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/thinking -> tampilkan tombol; /thinking <auto|low|medium|high|xhigh> -> set langsung."""
    user_id = update.effective_user.id
    if context.args:
        target = context.args[0].strip().lower()
        if target not in VALID_THINKING_MODES:
            valid = ", ".join(VALID_THINKING_MODES)
            await update.effective_message.reply_text(
                f"❌ Level tidak dikenali. Pilihan: <code>{html.escape(valid)}</code>",
                parse_mode=ParseMode.HTML,
            )
            return
        db.set_user_thinking_mode(user_id, target)

    active = db.get_user_thinking_mode(user_id)
    await update.effective_message.reply_text(
        format_thinking_text(active),
        parse_mode=ParseMode.HTML,
        reply_markup=build_thinking_keyboard(active),
    )


async def style_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menampilkan System Prompt dan Personality yang sedang aktif."""
    sys_prompt, personality, is_custom_p, is_custom_pers = get_effective_prompts()
    p_status = "Custom (DB Overwrite)" if is_custom_p else "Default (Hardcoded)"
    pers_status = "Custom (DB Overwrite)" if is_custom_pers else "Default (Hardcoded)"

    text = (
        f"🎭 <b>Konfigurasi System Prompt &amp; Personality</b>\n\n"
        f"📌 <b>System Prompt</b> [<i>{p_status}</i>]:\n"
        f"<code>{html.escape(sys_prompt)}</code>\n\n"
        f"🗣 <b>Personality</b> [<i>{pers_status}</i>]:\n"
        f"<code>{html.escape(personality)}</code>\n\n"
        f"🛠 <b>Cara Mengubah:</b>\n"
        f"• <code>/setprompt &lt;instruksi baru&gt;</code>\n"
        f"• <code>/setpersona &lt;gaya bicara baru&gt;</code>\n"
        f"• <code>/resetstyle</code> — Kembalikan keduanya ke default hardcoded"
    )
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)


async def setprompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menimpa (overwrite) DEFAULT_SYSTEM_PROMPT dan menyimpannya ke bot_config."""
    user_id = update.effective_user.id
    if not is_authorized_admin(user_id):
        await update.effective_message.reply_text(
            "⛔ Hanya Admin yang diizinkan mengubah System Prompt."
        )
        return

    raw_text = (update.effective_message.text or "").split(maxsplit=1)
    if len(raw_text) < 2 or not raw_text[1].strip():
        await update.effective_message.reply_text(
            "⚠️ Format: <code>/setprompt Kamu adalah senior Python architect...</code>\n"
            "Ketik /style untuk melihat System Prompt aktif saat ini.",
            parse_mode=ParseMode.HTML,
        )
        return

    new_prompt = raw_text[1].strip()
    db.set_config(CONFIG_KEY_SYSTEM_PROMPT, new_prompt)
    await update.effective_message.reply_text(
        f"✅ <b>System Prompt berhasil diperbarui (menimpa default)!</b>\n"
        f"<code>{html.escape(new_prompt)}</code>\n\n"
        f"<i>Berlaku untuk Gemini &amp; KoboldCPP. Gunakan /resetstyle untuk kembali ke default.</i>",
        parse_mode=ParseMode.HTML,
    )


async def setpersona_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menimpa (overwrite) DEFAULT_PERSONALITY dan menyimpannya ke bot_config."""
    user_id = update.effective_user.id
    if not is_authorized_admin(user_id):
        await update.effective_message.reply_text(
            "⛔ Hanya Admin yang diizinkan mengubah Personality."
        )
        return

    raw_text = (update.effective_message.text or "").split(maxsplit=1)
    if len(raw_text) < 2 or not raw_text[1].strip():
        await update.effective_message.reply_text(
            "⚠️ Format: <code>/setpersona Bicara santai, jenaka, pakai bahasa gaul Jakarta...</code>\n"
            "Ketik /style untuk melihat Personality aktif saat ini.",
            parse_mode=ParseMode.HTML,
        )
        return

    new_persona = raw_text[1].strip()
    db.set_config(CONFIG_KEY_PERSONALITY, new_persona)
    await update.effective_message.reply_text(
        f"✅ <b>Personality berhasil diperbarui (menimpa default)!</b>\n"
        f"<code>{html.escape(new_persona)}</code>\n\n"
        f"<i>Berlaku untuk Gemini &amp; KoboldCPP. Gunakan /resetstyle untuk kembali ke default.</i>",
        parse_mode=ParseMode.HTML,
    )


async def resetstyle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menghapus custom prompt & personality di DB agar kembali ke nilai hardcoded."""
    user_id = update.effective_user.id
    if not is_authorized_admin(user_id):
        await update.effective_message.reply_text(
            "⛔ Hanya Admin yang diizinkan mereset gaya bot."
        )
        return

    db.delete_config(CONFIG_KEY_SYSTEM_PROMPT)
    db.delete_config(CONFIG_KEY_PERSONALITY)
    await update.effective_message.reply_text(
        "🧹 <b>System Prompt &amp; Personality dikembalikan ke default hardcoded!</b>\n"
        "Ketik /style untuk memeriksa detailnya.",
        parse_mode=ParseMode.HTML,
    )


async def setproxy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_authorized_admin(user_id):
        await update.effective_message.reply_text(
            "⛔ Perintah ini hanya dapat dijalankan oleh Admin."
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
    user_id = update.effective_user.id
    if not is_authorized_admin(user_id):
        await update.effective_message.reply_text(
            "⛔ Perintah ini hanya dapat dijalankan oleh Admin."
        )
        return

    db.delete_config(CONFIG_KEY_PROXY_URL)
    user_mode = db.get_user_model_mode(user_id)
    await update.effective_message.reply_text(
        f"✅ <b>Proxy Dihapus!</b>\n"
        f"Bot kembali ke <b>Google AI Studio</b> (Mode Anda: <code>{html.escape(user_mode)}</code>).",
        parse_mode=ParseMode.HTML,
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    deleted = db.clear_chat_history(user_id)
    await update.effective_message.reply_text(
        f"🧹 Memori percakapan Anda direset ({deleted} pesan dihapus dari SQLite)."
    )


async def callback_query_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    if not query or not query.data:
        return

    user = query.from_user
    user_id = user.id
    data = query.data
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
            active_mode = db.get_user_model_mode(user_id)
            text = (
                f"🤖 <b>Pilih Mode / Model Gemini</b>\n"
                f"Mode aktif Anda: <code>{html.escape(active_mode)}</code>\n\n"
                f"Pilih <b>Auto Smart Router</b> untuk routing otomatis via <code>{DECISION_ROUTER_MODEL}</code>, "
                f"atau kunci salah satu model di bawah:"
            )
            await query.edit_message_text(
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=build_models_keyboard(active_mode),
            )

        elif data.startswith("setmodel:"):
            target_mode = data.split(":", 1)[1]
            if target_mode not in VALID_MODEL_MODES:
                await query.answer("❌ Mode/Model tidak valid.", show_alert=True)
                return

            db.set_user_model_mode(user_id, target_mode)
            text = (
                f"🤖 <b>Mode Model Diperbarui!</b>\n"
                f"Mode aktif Anda sekarang: <code>{html.escape(target_mode)}</code>"
            )
            await query.edit_message_text(
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=build_models_keyboard(target_mode),
            )

        elif data == "menu:thinking":
            active = db.get_user_thinking_mode(user_id)
            await query.edit_message_text(
                text=format_thinking_text(active),
                parse_mode=ParseMode.HTML,
                reply_markup=build_thinking_keyboard(active),
            )

        elif data.startswith("setthink:"):
            target = data.split(":", 1)[1]
            if target not in VALID_THINKING_MODES:
                await query.answer("❌ Level tidak valid.", show_alert=True)
                return
            db.set_user_thinking_mode(user_id, target)
            await query.edit_message_text(
                text=format_thinking_text(target),
                parse_mode=ParseMode.HTML,
                reply_markup=build_thinking_keyboard(target),
            )

        elif data == "menu:style":
            sys_prompt, personality, is_custom_p, is_custom_pers = get_effective_prompts()
            p_status = "Custom (DB)" if is_custom_p else "Hardcoded"
            pers_status = "Custom (DB)" if is_custom_pers else "Hardcoded"
            text = (
                f"🎭 <b>Prompt &amp; Personality Aktif</b>\n\n"
                f"📌 <b>System Prompt</b> [<i>{p_status}</i>]:\n"
                f"<code>{html.escape(sys_prompt[:400])}</code>\n\n"
                f"🗣 <b>Personality</b> [<i>{pers_status}</i>]:\n"
                f"<code>{html.escape(personality[:400])}</code>\n\n"
                f"Gunakan <code>/setprompt</code>, <code>/setpersona</code>, atau <code>/resetstyle</code>."
            )
            back_markup = InlineKeyboardMarkup(
                [[InlineKeyboardButton("⬅️ Kembali ke Menu", callback_data="menu:start")]]
            )
            await query.edit_message_text(
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=back_markup,
            )

        elif data == "menu:proxy_help":
            text = (
                "🔌 <b>Panduan Proxy KoboldCPP</b>\n\n"
                "Kirim perintah:\n"
                "<code>/setproxy https://nama-tunnel.trycloudflare.com</code>\n\n"
                "Sangat cocok jika Anda ingin roleplay/obrolan dengan gaya yang lebih bebas "
                "menggunakan LLM lokal Anda."
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
                await query.answer("⛔ Hanya Admin yang dapat menghapus proxy.", show_alert=True)
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


# ==============================================================================
# 9. ROUTER PESAN MASUK: TEKS, GAMBAR (PHOTO) & DOKUMEN (PDF / FILE KODE)
# ==============================================================================
async def _resolve_gemini_target_model(
    user_id: int,
    user_text: str,
    attachment_bytes: Optional[bytes] = None,
    attachment_mime: Optional[str] = None,
    attachment_name: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Menentukan (model_slug, thinking_level) untuk user:
    Smart Router 'auto' atau model manual; level thinking manual (low..xhigh)
    menimpa saran router, sedangkan 'auto' memakai saran router.
    """
    user_mode = db.get_user_model_mode(user_id)
    thinking_pref = db.get_user_thinking_mode(user_id)
    manual_level = thinking_pref if thinking_pref in THINKING_LEVELS else None

    if user_mode != SMART_ROUTER_MODE:
        return user_mode, manual_level

    route = await route_with_gemini_decision(
        user_text=user_text,
        attachment_bytes=attachment_bytes,
        attachment_mime=attachment_mime,
        attachment_name=attachment_name,
    )
    return route.model, (manual_level or route.thinking_level)


async def handle_text_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
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
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )
            target_model, thinking_level = await _resolve_gemini_target_model(
                user_id=user_id,
                user_text=user_text,
            )
            token_stream = stream_gemini_with_failover(
                model_slug=target_model,
                history=history,
                user_text=user_text,
                thinking_level=thinking_level,
            )

        final_reply = await stream_response_to_telegram(update, context, token_stream)
        db.add_chat_turn(user_id=user_id, user_text=user_text, assistant_text=final_reply)

    except requests.exceptions.Timeout:
        await update.effective_message.reply_text(
            "⏱ Koneksi ke KoboldCPP Proxy timeout. Pastikan PC lokal & tunnel aktif."
        )
    except requests.exceptions.RequestException:
        await update.effective_message.reply_text(
            "⚠️ Gagal terhubung ke KoboldCPP Proxy. Gunakan /clearproxy untuk kembali ke Gemini."
        )
    except genai_errors.APIError as exc:
        logger.error("Gemini APIError: %s", exc)
        await update.effective_message.reply_text(
            f"⚠️ Gemini API error (Kode {getattr(exc, 'code', 'Unknown')}). "
            "Coba ulangi atau pilih model manual lewat /models."
        )
    except Exception as exc:
        logger.exception("Error pesan teks user %s: %s", user_id, exc)
        await update.effective_message.reply_text(f"❌ Terjadi kesalahan: {exc}")


async def handle_photo_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    message = update.effective_message
    if not message or not message.photo:
        return

    user_id = update.effective_user.id
    caption = (message.caption or "").strip()
    prompt_text = caption if caption else "Analisis dan jelaskan gambar ini secara detail."

    if db.get_config(CONFIG_KEY_PROXY_URL):
        await message.reply_text(
            "ℹ️ Analisis gambar menggunakan <b>Google Gemini Vision</b>.\n"
            "Jalankan /clearproxy terlebih dahulu karena KoboldCPP Proxy sedang aktif.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
        tg_file = await message.photo[-1].get_file()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp_file:
            await tg_file.download_to_drive(custom_path=tmp_file.name)
            image_bytes = Path(tmp_file.name).read_bytes()

        target_model, thinking_level = await _resolve_gemini_target_model(
            user_id=user_id,
            user_text=prompt_text,
            attachment_bytes=image_bytes,
            attachment_mime="image/jpeg",
            attachment_name="photo.jpg",
        )
        history = db.get_chat_history(user_id=user_id, limit=MAX_HISTORY_MESSAGES)

        token_stream = stream_gemini_with_failover(
            model_slug=target_model,
            history=history,
            user_text=prompt_text,
            attachment_bytes=image_bytes,
            attachment_mime="image/jpeg",
            thinking_level=thinking_level,
        )

        final_reply = await stream_response_to_telegram(update, context, token_stream)
        db.add_chat_turn(
            user_id=user_id,
            user_text=f"[Mengirim Foto] {prompt_text}",
            assistant_text=final_reply,
        )
    except Exception as exc:
        logger.exception("Error memproses foto user %s: %s", user_id, exc)
        await message.reply_text(f"❌ Gagal memproses gambar: {exc}")


async def handle_document_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Mendukung pengiriman file PDF atau file teks/kode (misal .py, .txt, .json, .md)
    yang otomatis dievaluasi Smart Router (biasanya diarahkan ke HEAVY -> gemini-4-hydrogen).
    """
    message = update.effective_message
    if not message or not message.document:
        return

    doc = message.document
    user_id = update.effective_user.id
    file_name = doc.file_name or "document"
    mime_type = doc.mime_type or "application/octet-stream"
    caption = (message.caption or "").strip()
    prompt_text = caption if caption else f"Analisis isi file '{file_name}' ini secara menyeluruh."

    if db.get_config(CONFIG_KEY_PROXY_URL):
        await message.reply_text(
            "ℹ️ Analisis dokumen/PDF menggunakan <b>Google Gemini</b>.\n"
            "Jalankan /clearproxy terlebih dahulu karena KoboldCPP Proxy sedang aktif.",
            parse_mode=ParseMode.HTML,
        )
        return

    # Batasi ukuran file maksimal 15 MB agar hemat RAM container Pterodactyl
    if doc.file_size and doc.file_size > 15 * 1024 * 1024:
        await message.reply_text("⚠️ Ukuran file maksimal 15 MB.")
        return

    is_pdf = mime_type == "application/pdf" or file_name.lower().endswith(".pdf")
    is_text_or_code = mime_type.startswith("text/") or file_name.lower().endswith(
        (".py", ".js", ".ts", ".json", ".md", ".txt", ".csv", ".yaml", ".yml", ".html", ".css", ".sql")
    )

    if not (is_pdf or is_text_or_code):
        await message.reply_text(
            "⚠️ Format file belum didukung. Kirim file <b>PDF</b> atau file <b>teks/kode</b> (.py, .txt, .md, .json, dll).",
            parse_mode=ParseMode.HTML,
        )
        return

    if not is_pdf and doc.file_size and doc.file_size > MAX_TEXT_FILE_BYTES:
        await message.reply_text(
            f"⚠️ File teks/kode maksimal {MAX_TEXT_FILE_BYTES // (1024 * 1024)} MB. "
            "Kirim bagian yang relevan saja, atau gunakan PDF."
        )
        return

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
        tg_file = await doc.get_file()
        suffix = Path(file_name).suffix or (".pdf" if is_pdf else ".txt")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp_file:
            await tg_file.download_to_drive(custom_path=tmp_file.name)
            file_bytes = Path(tmp_file.name).read_bytes()

        history = db.get_chat_history(user_id=user_id, limit=MAX_HISTORY_MESSAGES)

        if is_pdf:
            target_model, thinking_level = await _resolve_gemini_target_model(
                user_id=user_id,
                user_text=prompt_text,
                attachment_bytes=file_bytes,
                attachment_mime="application/pdf",
                attachment_name=file_name,
            )
            token_stream = stream_gemini_with_failover(
                model_slug=target_model,
                history=history,
                user_text=prompt_text,
                attachment_bytes=file_bytes,
                attachment_mime="application/pdf",
                thinking_level=thinking_level,
            )
        else:
            # Untuk file kode/teks, sisipkan langsung sebagai blok kode teks
            if len(file_bytes) > MAX_TEXT_FILE_BYTES:
                await message.reply_text(
                    f"⚠️ File teks/kode maksimal {MAX_TEXT_FILE_BYTES // (1024 * 1024)} MB. "
                    "Kirim bagian yang relevan saja, atau gunakan PDF."
                )
                return
            decoded_text = file_bytes.decode("utf-8", errors="replace")
            combined_text = (
                f"{prompt_text}\n\n"
                f"--- ISI FILE: {file_name} ---\n"
                f"{decoded_text}\n"
                f"--- AKHIR FILE ---"
            )
            # Router cukup melihat pesan + cuplikan awal file, bukan seluruh isinya
            router_preview = (
                f"{prompt_text}\n\n[File: {file_name}, {len(file_bytes)} bytes]\n"
                f"{decoded_text[:2000]}"
            )
            target_model, thinking_level = await _resolve_gemini_target_model(
                user_id=user_id,
                user_text=router_preview,
                attachment_name=file_name,
            )
            token_stream = stream_gemini_with_failover(
                model_slug=target_model,
                history=history,
                user_text=combined_text,
                thinking_level=thinking_level,
            )

        final_reply = await stream_response_to_telegram(update, context, token_stream)
        db.add_chat_turn(
            user_id=user_id,
            user_text=f"[Mengirim File: {file_name}] {prompt_text}",
            assistant_text=final_reply,
        )
    except Exception as exc:
        logger.exception("Error memproses dokumen user %s: %s", user_id, exc)
        await message.reply_text(f"❌ Gagal memproses file: {exc}")


async def global_error_handler(
    update: object, context: ContextTypes.DEFAULT_TYPE
) -> None:
    logger.exception("Unhandled Telegram Bot error: %s", context.error)


async def post_init_setup(application: Application) -> None:
    commands = [
        BotCommand("start", "Menu utama, status backend & tombol cepat"),
        BotCommand("models", "Pilih Auto Smart Router atau model Gemini"),
        BotCommand("setmodel", "Set mode model (/setmodel auto | <slug>)"),
        BotCommand("thinking", "Level berpikir Gemini 4 (/thinking auto|low|medium|high|xhigh)"),
        BotCommand("style", "Lihat System Prompt & Personality aktif"),
        BotCommand("setprompt", "Timpa System Prompt default (/setprompt <teks>)"),
        BotCommand("setpersona", "Timpa Personality default (/setpersona <teks>)"),
        BotCommand("resetstyle", "Kembalikan System Prompt & Personality ke default"),
        BotCommand("setproxy", "Aktifkan KoboldCPP Proxy (/setproxy <url>)"),
        BotCommand("clearproxy", "Hapus proxy & kembali ke Gemini"),
        BotCommand("reset", "Hapus riwayat memori chat Anda"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Menu perintah Telegram berhasil disinkronkan.")


# ==============================================================================
# 10. ENTRYPOINT APLIKASI
# ==============================================================================
def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            f"TELEGRAM_BOT_TOKEN tidak ditemukan di {ENV_PATH}."
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
    application.add_handler(CommandHandler("thinking", thinking_command))
    application.add_handler(CommandHandler("style", style_command))
    application.add_handler(CommandHandler("setprompt", setprompt_command))
    application.add_handler(CommandHandler("setpersona", setpersona_command))
    application.add_handler(CommandHandler("resetstyle", resetstyle_command))
    application.add_handler(CommandHandler("setproxy", setproxy_command))
    application.add_handler(CommandHandler("clearproxy", clearproxy_command))
    application.add_handler(CommandHandler("reset", reset_command))

    # Inline Keyboard Callback Handler
    application.add_handler(CallbackQueryHandler(callback_query_handler))

    # Multimodal Handlers (Foto, Dokumen PDF/Kode) & Teks
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document_message))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message)
    )

    application.add_error_handler(global_error_handler)

    logger.info("Bot Dual-Backend + Smart Router aktif | Base Dir: %s", BASE_DIR)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
