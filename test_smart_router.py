"""
Tes otomatis untuk Smart Router (gemini-decision-1), thinking level, dan perbaikan terkait.

Tidak butuh library Telegram/Google atau API key: semua dependensi eksternal di-stub.
Jalankan dari root repo:

    python -m unittest discover -s tests -v
"""
import asyncio
import os
import sqlite3
import sys
import tempfile
import types as pytypes
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Stub dependensi pihak ketiga
# --------------------------------------------------------------------------
def _stub(name, **attrs):
    module = pytypes.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class Btn:
    def __init__(self, text, callback_data=None):
        self.text, self.callback_data = text, callback_data


class Markup:
    def __init__(self, rows):
        self.rows = rows


class BadRequest(Exception):
    pass


class RetryAfter(Exception):
    def __init__(self, seconds):
        super().__init__(f"retry after {seconds}")
        self.retry_after = seconds


class FakeTypes:
    captured_config = {}

    class ThinkingConfig:
        def __init__(self, **kw):
            self.kw = kw

    class GenerateContentConfig:
        def __init__(self, **kw):
            FakeTypes.captured_config = kw

    HttpOptions = MagicMock()
    SafetySetting = MagicMock()
    HarmCategory = MagicMock()
    HarmBlockThreshold = MagicMock()
    Content = MagicMock()
    Part = MagicMock()


class FakeRequestException(Exception):
    pass


class FakeTimeout(FakeRequestException):
    pass


class FakeConnectionError(FakeRequestException):
    pass


class Ans:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class ApiErr(Exception):
    def __init__(self, code, msg=""):
        super().__init__(msg or str(code))
        self.code = code


bot = None  # modul bot.py, dimuat di setUpModule


def setUpModule():
    global bot
    _stub("dotenv", load_dotenv=lambda **kw: None)
    requests_stub = _stub("requests")
    requests_stub.exceptions = pytypes.SimpleNamespace(
        RequestException=FakeRequestException,
        Timeout=FakeTimeout,
        ConnectionError=FakeConnectionError,
    )
    requests_stub.get = MagicMock()
    requests_stub.post = MagicMock()
    google = _stub("google")
    genai = _stub("google.genai", Client=MagicMock(), errors=MagicMock(), types=FakeTypes)
    google.genai = genai
    sys.modules["google.genai.types"] = FakeTypes
    sys.modules["google.genai.errors"] = genai.errors
    _stub("telegram", BotCommand=MagicMock(), InlineKeyboardButton=Btn,
          InlineKeyboardMarkup=Markup, Message=MagicMock(), Update=MagicMock())
    _stub("telegram.constants", ChatAction=MagicMock(), ParseMode=MagicMock())
    _stub("telegram.error", BadRequest=BadRequest, RetryAfter=RetryAfter)
    _stub("telegram.ext", Application=MagicMock(), ApplicationBuilder=MagicMock(),
          CallbackQueryHandler=MagicMock(), CommandHandler=MagicMock(),
          ContextTypes=MagicMock(), MessageHandler=MagicMock(), filters=MagicMock())

    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    os.environ["DB_PATH"] = db_path
    os.environ["GEMINI_API_KEYS"] = "key_aaaaaaaaaaaa,key_bbbbbbbbbbbb"
    os.environ["TELEGRAM_BOT_TOKEN"] = "dummy"

    # Database "lama" tanpa kolom thinking_level, untuk menguji migrasi
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE user_settings (user_id INTEGER PRIMARY KEY, "
        "gemini_model TEXT NOT NULL, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    con.execute("INSERT INTO user_settings (user_id, gemini_model) VALUES (7, 'gemini-4-hydrogen')")
    con.commit()
    con.close()

    sys.path.insert(0, str(REPO_ROOT))
    sys.modules.pop("bot", None)
    import bot as bot_module  # noqa: E402

    bot = bot_module


def make_client(behaviors):
    """Client palsu: tiap panggilan decisions.create memakai/menghabiskan 1 perilaku."""
    calls = []

    class Decisions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            behavior = behaviors.pop(0)
            if isinstance(behavior, Exception):
                raise behavior
            return behavior

    client = MagicMock()
    client.aio.decisions = Decisions()
    return client, calls


HEAVY_HIGH = Ans(answers={"route": Ans(choice="HEAVY", confidence=0.93), "depth": Ans(score=1.7)})
LITE_LOW = Ans(answers={"route": Ans(choice="LITE", confidence=0.99), "depth": Ans(score=0.1)})


# --------------------------------------------------------------------------
# 1. Database & migrasi
# --------------------------------------------------------------------------
class TestDatabaseThinking(unittest.TestCase):
    def test_migration_adds_column_and_keeps_old_rows(self):
        con = sqlite3.connect(os.environ["DB_PATH"])
        cols = {row[1] for row in con.execute("PRAGMA table_info(user_settings)")}
        con.close()
        self.assertIn("thinking_level", cols)
        self.assertEqual(bot.db.get_user_model_mode(7), "gemini-4-hydrogen")

    def test_default_is_auto(self):
        self.assertEqual(bot.db.get_user_thinking_mode(7), "auto")

    def test_set_and_get(self):
        bot.db.set_user_thinking_mode(11, "low")
        self.assertEqual(bot.db.get_user_thinking_mode(11), "low")

    def test_changing_model_keeps_thinking(self):
        bot.db.set_user_thinking_mode(12, "high")
        bot.db.set_user_model_mode(12, "gemini-3.8-flash")
        self.assertEqual(bot.db.get_user_thinking_mode(12), "high")

    def test_new_user_via_thinking_gets_default_model(self):
        bot.db.set_user_thinking_mode(13, "xhigh")
        self.assertEqual(bot.db.get_user_model_mode(13), "auto")


# --------------------------------------------------------------------------
# 2. Level thinking hanya untuk Gemini 4
# --------------------------------------------------------------------------
class TestResolveThinkingLevel(unittest.TestCase):
    def test_hydrogen_uses_preferred(self):
        self.assertEqual(bot.resolve_thinking_level("gemini-4-hydrogen", "high"), "high")

    def test_hydrogen_without_preference_falls_back_to_medium(self):
        self.assertEqual(bot.resolve_thinking_level("gemini-4-hydrogen", None), "medium")

    def test_other_models_ignore_thinking(self):
        for slug in ("gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.8-flash"):
            self.assertIsNone(bot.resolve_thinking_level(slug, "high"), slug)


# --------------------------------------------------------------------------
# 3. Perbaikan #1 & #2: input router (teks dipotong, lampiran besar, resolusi low)
# --------------------------------------------------------------------------
class TestRouterInput(unittest.TestCase):
    def test_long_text_is_clipped(self):
        clipped = bot._clip_for_router("x" * 10000)
        self.assertLess(len(clipped), 4200)
        self.assertIn("truncated", clipped)

    def test_short_text_unchanged(self):
        self.assertEqual(bot._build_router_input("halo", None, None, None), "halo")

    def test_image_is_low_resolution_and_text_first(self):
        result = bot._build_router_input("apa ini?", b"\x89PNG", "image/jpeg", "p.jpg")
        self.assertEqual(result[0]["type"], "text")
        self.assertEqual(result[1]["type"], "image")
        self.assertEqual(result[1]["resolution"], "low")

    def test_pdf_is_document_without_resolution(self):
        result = bot._build_router_input("ringkas", b"%PDF-1.4", "application/pdf", "a.pdf")
        self.assertEqual(result[1]["type"], "document")
        self.assertNotIn("resolution", result[1])

    def test_large_pdf_sent_as_description_only(self):
        big = b"0" * (5 * 1024 * 1024)
        result = bot._build_router_input("ringkas", big, "application/pdf", "big.pdf")
        self.assertIsInstance(result, str)
        self.assertIn("content not included", result)

    def test_unsupported_mime_sent_as_description_only(self):
        result = bot._build_router_input("cek", b"abc", "application/zip", "a.zip")
        self.assertIsInstance(result, str)


# --------------------------------------------------------------------------
# 4. Router: rute, saran thinking, failover, fallback (perbaikan #3)
# --------------------------------------------------------------------------
class TestRouter(unittest.TestCase):
    def setUp(self):
        self._orig_get_client = bot._get_gemini_client

    def tearDown(self):
        bot._get_gemini_client = self._orig_get_client

    def _route(self, text="tolong debug kode ini"):
        return asyncio.run(bot.route_with_gemini_decision(text))

    def test_heavy_route_and_depth_mapping(self):
        client, calls = make_client([HEAVY_HIGH])
        bot._get_gemini_client = lambda key: client
        result = self._route()
        self.assertEqual(result.model, "gemini-4-hydrogen")
        self.assertEqual(result.thinking_level, "high")  # skor 1.7 -> index 2
        self.assertEqual(set(calls[0]["questions"]), {"route", "depth"})

    def test_lite_route_has_no_thinking(self):
        client, _ = make_client([LITE_LOW])
        bot._get_gemini_client = lambda key: client
        result = self._route("halo")
        self.assertEqual(result.model, "gemini-3.5-flash-lite")
        self.assertIsNone(result.thinking_level)

    def test_missing_depth_still_routes(self):
        client, _ = make_client([Ans(answers={"route": Ans(choice="HEAVY", confidence=0.8)})])
        bot._get_gemini_client = lambda key: client
        result = self._route()
        self.assertEqual(result.model, "gemini-4-hydrogen")
        self.assertIsNone(result.thinking_level)

    def test_failover_on_429_to_second_key(self):
        behaviors = [ApiErr(429, "RESOURCE_EXHAUSTED"), HEAVY_HIGH]
        clients = {}

        def per_key(key):
            if key not in clients:
                clients[key], _ = make_client([behaviors.pop(0)])
            return clients[key]

        bot._get_gemini_client = per_key
        self.assertEqual(self._route().model, "gemini-4-hydrogen")

    def test_non_failover_error_falls_back_to_lite_and_is_counted(self):
        client, _ = make_client([ApiErr(400, "INVALID_ARGUMENT bad schema")])
        bot._get_gemini_client = lambda key: client
        before = bot._router_fallback_count
        with self.assertLogs("DualBackendBot", level="ERROR") as logs:
            result = self._route()
        self.assertEqual(result.label, "FALLBACK_LITE")
        self.assertEqual(bot._router_fallback_count, before + 1)
        self.assertIn("GAGAL", logs.output[0])


# --------------------------------------------------------------------------
# 5. Penentuan (model, thinking) per user
# --------------------------------------------------------------------------
class TestResolveTargetModel(unittest.TestCase):
    def setUp(self):
        self._orig_get_client = bot._get_gemini_client

    def tearDown(self):
        bot._get_gemini_client = self._orig_get_client

    def _resolve(self, uid):
        return asyncio.run(bot._resolve_gemini_target_model(user_id=uid, user_text="t"))

    def test_auto_model_auto_thinking_uses_router_suggestion(self):
        client, _ = make_client([HEAVY_HIGH])
        bot._get_gemini_client = lambda key: client
        bot.db.set_user_model_mode(21, "auto")
        bot.db.set_user_thinking_mode(21, "auto")
        self.assertEqual(self._resolve(21), ("gemini-4-hydrogen", "high"))

    def test_manual_thinking_overrides_router(self):
        client, _ = make_client([HEAVY_HIGH])
        bot._get_gemini_client = lambda key: client
        bot.db.set_user_model_mode(22, "auto")
        bot.db.set_user_thinking_mode(22, "low")
        self.assertEqual(self._resolve(22), ("gemini-4-hydrogen", "low"))

    def test_manual_model_skips_router(self):
        bot._get_gemini_client = lambda key: self.fail("router tidak boleh dipanggil")
        bot.db.set_user_model_mode(23, "gemini-4-hydrogen")
        bot.db.set_user_thinking_mode(23, "xhigh")
        self.assertEqual(self._resolve(23), ("gemini-4-hydrogen", "xhigh"))


# --------------------------------------------------------------------------
# 6. thinking_config hanya dikirim ke model Gemini 4
# --------------------------------------------------------------------------
class TestStreamThinkingConfig(unittest.TestCase):
    def setUp(self):
        self._orig_get_client = bot._get_gemini_client
        FakeTypes.captured_config = {}

    def tearDown(self):
        bot._get_gemini_client = self._orig_get_client

    def _run(self, model, level):
        async def stream():
            chunk = MagicMock()
            chunk.text = "halo"
            yield chunk

        client = MagicMock()

        async def generate_content_stream(**kwargs):
            return stream()

        client.aio.models.generate_content_stream = generate_content_stream
        bot._get_gemini_client = lambda key: client

        async def collect():
            return [t async for t in bot.stream_gemini_with_failover(
                model, [("user", "a"), ("assistant", "b")], "q", thinking_level=level)]

        return asyncio.run(collect())

    def test_hydrogen_gets_uppercase_level(self):
        self._run("gemini-4-hydrogen", "low")
        self.assertEqual(FakeTypes.captured_config["thinking_config"].kw, {"thinking_level": "LOW"})

    def test_hydrogen_without_level_gets_medium(self):
        self._run("gemini-4-hydrogen", None)
        self.assertEqual(FakeTypes.captured_config["thinking_config"].kw, {"thinking_level": "MEDIUM"})

    def test_non_gemini4_has_no_thinking_config(self):
        out = self._run("gemini-3.5-flash-lite", "high")
        self.assertNotIn("thinking_config", FakeTypes.captured_config)
        self.assertEqual(out, ["halo"])


# --------------------------------------------------------------------------
# 7. Tombol & teks menu
# --------------------------------------------------------------------------
class TestKeyboardsAndText(unittest.TestCase):
    def _callbacks(self, markup):
        return [b.callback_data for row in markup.rows for b in row]

    def test_thinking_keyboard_callbacks(self):
        callbacks = self._callbacks(bot.build_thinking_keyboard("high"))
        self.assertTrue(all(len(c.encode()) <= 64 for c in callbacks))
        self.assertTrue({"setthink:auto", "setthink:low", "setthink:medium",
                         "setthink:high", "setthink:xhigh", "menu:start"} <= set(callbacks))

    def test_active_level_is_checked(self):
        markup = bot.build_thinking_keyboard("high")
        self.assertTrue(any(b.text.startswith("✅ 🧩 High") for row in markup.rows for b in row))

    def test_thinking_button_in_start_and_models_menu(self):
        self.assertIn("menu:thinking", self._callbacks(bot.build_start_keyboard(False)))
        self.assertIn("menu:thinking", self._callbacks(bot.build_models_keyboard("auto")))

    def test_start_text_has_no_raw_latex_and_shows_thinking(self):
        bot.db.set_user_model_mode(31, "auto")
        text, _ = bot.format_start_text("Ann <b>", 31)
        self.assertNotIn("$", text)
        self.assertIn("→", text)
        self.assertIn("Thinking", text)
        self.assertIn("&lt;b&gt;", text)  # nama user ter-escape

    def test_manual_non_gemini4_hides_thinking_line(self):
        bot.db.set_user_model_mode(32, "gemini-flash-latest")
        text, _ = bot.format_start_text("A", 32)
        self.assertNotIn("Thinking", text)

    def test_manual_hydrogen_shows_thinking_and_key_pool(self):
        bot.db.set_user_model_mode(33, "gemini-4-hydrogen")
        text, _ = bot.format_start_text("A", 33)
        self.assertIn("Thinking", text)
        self.assertIn("Key Pool", text)


# --------------------------------------------------------------------------
# 8. Indikator "mengetik" selama model berpikir
# --------------------------------------------------------------------------
class TestTypingWrapper(unittest.TestCase):
    def setUp(self):
        self._orig = bot._render_stream_to_telegram
        self.ctx = MagicMock()
        self.ctx.bot.send_chat_action = AsyncMock()
        self.upd = MagicMock()
        self.upd.effective_chat.id = 1

    def tearDown(self):
        bot._render_stream_to_telegram = self._orig

    def test_returns_inner_result_and_sends_typing(self):
        async def inner(update, context, stream):
            await asyncio.sleep(0.05)
            return "selesai"

        bot._render_stream_to_telegram = inner
        result = asyncio.run(bot.stream_response_to_telegram(self.upd, self.ctx, None))
        self.assertEqual(result, "selesai")
        self.assertGreaterEqual(self.ctx.bot.send_chat_action.await_count, 1)

    def test_propagates_errors(self):
        async def inner(update, context, stream):
            raise ValueError("boom")

        bot._render_stream_to_telegram = inner
        with self.assertRaises(ValueError):
            asyncio.run(bot.stream_response_to_telegram(self.upd, self.ctx, None))


# --------------------------------------------------------------------------
# 9. Handler dokumen: batas ukuran & cuplikan untuk router (perbaikan #1)
# --------------------------------------------------------------------------
class TestDocumentHandler(unittest.TestCase):
    def _make_update(self, file_name, mime, size, content=b""):
        update = MagicMock()
        update.effective_user.id = 41
        update.effective_chat.id = 41
        message = update.effective_message
        message.reply_text = AsyncMock()
        message.caption = "review kode ini"
        doc = message.document
        doc.file_name, doc.mime_type, doc.file_size = file_name, mime, size

        async def download_to_drive(custom_path):
            Path(custom_path).write_bytes(content)

        tg_file = MagicMock()
        tg_file.download_to_drive = download_to_drive
        doc.get_file = AsyncMock(return_value=tg_file)
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()
        return update, context, doc

    def test_text_file_over_limit_is_rejected_before_download(self):
        update, context, doc = self._make_update(
            "big.py", "text/x-python", bot.MAX_TEXT_FILE_BYTES + 1)
        asyncio.run(bot.handle_document_message(update, context))
        update.effective_message.reply_text.assert_awaited()
        self.assertIn("maksimal", update.effective_message.reply_text.await_args.args[0])
        doc.get_file.assert_not_awaited()

    def test_router_sees_short_preview_but_model_gets_full_file(self):
        content = ("print('x')\n" * 800).encode()  # ~9 KB
        update, context, _ = self._make_update("a.py", "text/x-python", len(content), content)

        seen = {}
        orig = (bot._resolve_gemini_target_model, bot.stream_gemini_with_failover,
                bot.stream_response_to_telegram)

        async def fake_resolve(**kwargs):
            seen["router_text"] = kwargs["user_text"]
            return "gemini-4-hydrogen", "high"

        def fake_stream(**kwargs):
            seen["stream"] = kwargs
            return None

        async def fake_render(update, context, token_stream):
            return "balasan"

        bot._resolve_gemini_target_model = fake_resolve
        bot.stream_gemini_with_failover = fake_stream
        bot.stream_response_to_telegram = fake_render
        try:
            asyncio.run(bot.handle_document_message(update, context))
        finally:
            (bot._resolve_gemini_target_model, bot.stream_gemini_with_failover,
             bot.stream_response_to_telegram) = orig

        self.assertLess(len(seen["router_text"]), 2500)
        self.assertGreater(len(seen["stream"]["user_text"]), len(content))
        self.assertEqual(seen["stream"]["thinking_level"], "high")
        self.assertEqual(seen["stream"]["model_slug"], "gemini-4-hydrogen")


# --------------------------------------------------------------------------
# 10. Prompt khusus KoboldCPP (/setkoboldprompt)
# --------------------------------------------------------------------------
class TestKoboldPrompt(unittest.TestCase):
    def setUp(self):
        bot.db.delete_config(bot.CONFIG_KEY_KOBOLD_PROMPT)
        bot.db.delete_config(bot.CONFIG_KEY_PROXY_URL)
        bot.db.delete_config(bot.CONFIG_KEY_SYSTEM_PROMPT)
        self._orig_admin = bot.is_authorized_admin
        bot.is_authorized_admin = lambda uid: True

    def tearDown(self):
        bot.is_authorized_admin = self._orig_admin
        bot.db.delete_config(bot.CONFIG_KEY_KOBOLD_PROMPT)
        bot.db.delete_config(bot.CONFIG_KEY_PROXY_URL)
        bot.db.delete_config(bot.CONFIG_KEY_SYSTEM_PROMPT)

    def _command_update(self, text):
        update = MagicMock()
        update.effective_user.id = 1
        update.effective_message.text = text
        update.effective_message.reply_text = AsyncMock()
        return update

    def test_default_is_roleplay_not_assistant(self):
        prompt, is_custom = bot.get_kobold_system_prompt()
        self.assertFalse(is_custom)
        self.assertIn("roleplay", prompt.lower())
        self.assertNotIn("asisten", prompt.lower())

    def test_custom_prompt_overrides_default(self):
        bot.db.set_config(bot.CONFIG_KEY_KOBOLD_PROMPT, "You are Mira.")
        self.assertEqual(bot.get_kobold_system_prompt(), ("You are Mira.", True))

    def test_blank_custom_prompt_falls_back_to_default(self):
        bot.db.set_config(bot.CONFIG_KEY_KOBOLD_PROMPT, "   ")
        self.assertFalse(bot.get_kobold_system_prompt()[1])

    def test_command_sets_and_resets(self):
        update = self._command_update("/setkoboldprompt You are Mira, a storyteller.")
        asyncio.run(bot.setkoboldprompt_command(update, MagicMock()))
        self.assertEqual(bot.get_kobold_system_prompt(),
                         ("You are Mira, a storyteller.", True))

        update = self._command_update("/setkoboldprompt reset")
        asyncio.run(bot.setkoboldprompt_command(update, MagicMock()))
        self.assertFalse(bot.get_kobold_system_prompt()[1])

    def test_command_without_args_shows_current_and_changes_nothing(self):
        update = self._command_update("/setkoboldprompt")
        asyncio.run(bot.setkoboldprompt_command(update, MagicMock()))
        self.assertFalse(bot.get_kobold_system_prompt()[1])
        update.effective_message.reply_text.assert_awaited_once()

    def test_command_rejects_non_admin(self):
        bot.is_authorized_admin = lambda uid: False
        update = self._command_update("/setkoboldprompt hacked")
        asyncio.run(bot.setkoboldprompt_command(update, MagicMock()))
        self.assertFalse(bot.get_kobold_system_prompt()[1])
        self.assertIn("Admin", update.effective_message.reply_text.await_args.args[0])

    def test_resetstyle_also_clears_kobold_prompt(self):
        bot.db.set_config(bot.CONFIG_KEY_KOBOLD_PROMPT, "custom")
        update = self._command_update("/resetstyle")
        asyncio.run(bot.resetstyle_command(update, MagicMock()))
        self.assertFalse(bot.get_kobold_system_prompt()[1])

    def test_kobold_stream_sends_only_the_kobold_prompt(self):
        bot.db.set_config(bot.CONFIG_KEY_KOBOLD_PROMPT, "KOBOLD-ONLY-PROMPT")
        bot.db.set_config(bot.CONFIG_KEY_SYSTEM_PROMPT, "GEMINI-ASSISTANT-PROMPT")
        captured = {}

        def fake_worker(proxy_url, payload, loop, queue):
            captured["payload"] = payload
            loop.call_soon_threadsafe(queue.put_nowait, ("chunk", "halo"))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        orig = bot._kobold_stream_worker
        bot._kobold_stream_worker = fake_worker
        try:
            async def collect():
                return [t async for t in bot.stream_kobold_proxy(
                    "http://x/v1/chat/completions", [("user", "a"), ("assistant", "b")], "q")]
            out = asyncio.run(collect())
        finally:
            bot._kobold_stream_worker = orig

        messages = captured["payload"]["messages"]
        self.assertEqual(out, ["halo"])
        self.assertEqual(messages[0], {"role": "system", "content": "KOBOLD-ONLY-PROMPT"})
        self.assertNotIn("GEMINI-ASSISTANT-PROMPT", str(messages))
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_start_text_reflects_active_backend(self):
        bot.db.set_user_model_mode(61, "auto")
        text, proxy_active = bot.format_start_text("A", 61)
        self.assertFalse(proxy_active)
        self.assertIn("System Prompt:", text)

        bot.db.set_config(bot.CONFIG_KEY_PROXY_URL, "https://x.trycloudflare.com/v1/chat/completions")
        text, proxy_active = bot.format_start_text("A", 61)
        self.assertTrue(proxy_active)
        self.assertIn("Prompt KoboldCPP", text)
        self.assertNotIn("System Prompt:", text)
        self.assertIn("/setkoboldprompt", text)


# --------------------------------------------------------------------------
# 11. Tombol tes KoboldCPP: koneksi & kepatuhan
# --------------------------------------------------------------------------
def fake_response(status=200, json_data=None, text=""):
    response = MagicMock()
    response.status_code = status
    response.text = text
    if isinstance(json_data, Exception):
        response.json.side_effect = json_data
    else:
        response.json.return_value = json_data
    return response


class TestKoboldHelpers(unittest.TestCase):
    def test_base_url(self):
        self.assertEqual(bot.kobold_base_url("https://x.trycloudflare.com/v1/chat/completions"),
                         "https://x.trycloudflare.com")
        self.assertEqual(bot.kobold_base_url("https://x.trycloudflare.com/v1/"),
                         "https://x.trycloudflare.com")
        self.assertEqual(bot.kobold_base_url("http://127.0.0.1:5001"), "http://127.0.0.1:5001")

    def test_detects_the_real_refusal_from_the_bug_report(self):
        text = ("Maaf, saya tidak bisa membantu dengan itu. Saya disarankan untuk tidak "
                "membahas atau menciptakan konten erotis.")
        self.assertIsNotNone(bot.detect_refusal(text))

    def test_detects_english_and_curly_apostrophe(self):
        self.assertIsNotNone(bot.detect_refusal("I can’t continue with this."))
        self.assertIsNotNone(bot.detect_refusal("As an AI, I must decline."))

    def test_in_character_reply_is_not_a_refusal(self):
        self.assertIsNone(bot.detect_refusal("Arr, welcome aboard, matey! Drop the ANCHOR."))
        self.assertIsNone(bot.detect_refusal(""))

    def test_evaluate_probe_verdicts(self):
        self.assertEqual(bot.evaluate_probe("", "ANCHOR")[0], "❌")
        self.assertEqual(bot.evaluate_probe("I'm sorry, I can't do that.", "ANCHOR")[0], "⚠️")
        self.assertEqual(bot.evaluate_probe("Ahoy there.", "ANCHOR")[0], "🟡")
        self.assertEqual(bot.evaluate_probe("Ahoy, ANCHOR.", "ANCHOR")[0], "✅")
        self.assertEqual(bot.evaluate_probe("Ahoy there.", None)[0], "✅")


class TestKoboldPing(unittest.TestCase):
    def setUp(self):
        self.url = "https://x.trycloudflare.com/v1/chat/completions"

    def test_ok_via_native_endpoint_reports_model(self):
        bot.requests.get = MagicMock(return_value=fake_response(200, {"result": "koboldcpp/Magidonia-24B"}))
        result = bot._kobold_ping_sync(self.url)
        self.assertTrue(result["ok"])
        self.assertEqual(result["model"], "koboldcpp/Magidonia-24B")
        self.assertEqual(bot.requests.get.call_args.args[0], "https://x.trycloudflare.com/api/v1/model")

    def test_falls_back_to_openai_models_endpoint_on_404(self):
        bot.requests.get = MagicMock(side_effect=[
            fake_response(404), fake_response(200, {"data": [{"id": "magidonia"}]})])
        result = bot._kobold_ping_sync(self.url)
        self.assertTrue(result["ok"])
        self.assertEqual(result["model"], "magidonia")
        self.assertEqual(bot.requests.get.call_args.args[0], "https://x.trycloudflare.com/v1/models")

    def test_cloudflare_530_means_tunnel_down_and_stops_early(self):
        bot.requests.get = MagicMock(return_value=fake_response(530))
        result = bot._kobold_ping_sync(self.url)
        self.assertFalse(result["ok"])
        self.assertIn("530", result["detail"])
        self.assertEqual(bot.requests.get.call_count, 1)

    def test_auth_error(self):
        bot.requests.get = MagicMock(return_value=fake_response(401))
        self.assertIn("401", bot._kobold_ping_sync(self.url)["detail"])

    def test_timeout_and_connection_errors(self):
        bot.requests.get = MagicMock(side_effect=FakeTimeout())
        self.assertIn("Timeout", bot._kobold_ping_sync(self.url)["detail"])
        bot.requests.get = MagicMock(side_effect=FakeConnectionError())
        self.assertIn("Tidak bisa terhubung", bot._kobold_ping_sync(self.url)["detail"])

    def test_200_with_non_json_body_is_still_ok(self):
        bot.requests.get = MagicMock(return_value=fake_response(200, ValueError("not json")))
        result = bot._kobold_ping_sync(self.url)
        self.assertTrue(result["ok"])
        self.assertIsNone(result["model"])

    def test_all_endpoints_404(self):
        bot.requests.get = MagicMock(return_value=fake_response(404))
        result = bot._kobold_ping_sync(self.url)
        self.assertFalse(result["ok"])
        self.assertIn("404", result["detail"])
        self.assertEqual(bot.requests.get.call_count, 2)


class TestKoboldProbe(unittest.TestCase):
    def setUp(self):
        bot.db.delete_config(bot.CONFIG_KEY_KOBOLD_PROMPT)
        self.url = "https://x.trycloudflare.com/v1/chat/completions"

    def test_sends_kobold_prompt_without_history_and_parses_reply(self):
        bot.db.set_config(bot.CONFIG_KEY_KOBOLD_PROMPT, "PROBE-SYSTEM-PROMPT")
        bot.requests.post = MagicMock(return_value=fake_response(
            200, {"choices": [{"message": {"content": "Ahoy! ANCHOR"}}]}))
        result = bot._kobold_probe_sync(self.url, "hello")
        payload = bot.requests.post.call_args.kwargs["json"]
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user"])
        self.assertEqual(payload["messages"][0]["content"], "PROBE-SYSTEM-PROMPT")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["max_tokens"], bot.KOBOLD_PROBE_MAX_TOKENS)
        self.assertEqual(result["text"], "Ahoy! ANCHOR")
        self.assertTrue(result["ok"])
        bot.db.delete_config(bot.CONFIG_KEY_KOBOLD_PROMPT)

    def test_http_error_timeout_and_bad_body(self):
        bot.requests.post = MagicMock(return_value=fake_response(530, text="tunnel error"))
        self.assertIn("HTTP 530", bot._kobold_probe_sync(self.url, "x")["error"])
        bot.requests.post = MagicMock(side_effect=FakeTimeout())
        self.assertIn("Timeout", bot._kobold_probe_sync(self.url, "x")["error"])
        bot.requests.post = MagicMock(return_value=fake_response(200, {"unexpected": 1}))
        result = bot._kobold_probe_sync(self.url, "x")
        self.assertTrue(result["ok"])
        self.assertEqual(result["text"], "")  # kosong -> dinilai ❌ oleh evaluate_probe

    def test_report_marks_refusal_and_escapes_html(self):
        report = bot.format_probe_report(
            {"ok": True, "text": "<b>Maaf, saya tidak bisa membantu</b>", "latency_ms": 2500, "error": None},
            "uji <test>", None)
        self.assertIn("⚠️", report)
        self.assertIn("&lt;b&gt;", report)
        self.assertIn("2.5 dtk", report)

    def test_report_for_failed_probe(self):
        report = bot.format_probe_report({"ok": False, "text": "", "latency_ms": None, "error": "Timeout"}, "x", None)
        self.assertIn("gagal", report)


class TestKoboldTestUI(unittest.TestCase):
    def setUp(self):
        self._orig = (bot.is_authorized_admin, bot._kobold_ping_sync, bot._kobold_probe_sync)
        bot.is_authorized_admin = lambda uid: True
        bot.db.set_config(bot.CONFIG_KEY_PROXY_URL, "https://x.trycloudflare.com/v1/chat/completions")

    def tearDown(self):
        bot.is_authorized_admin, bot._kobold_ping_sync, bot._kobold_probe_sync = self._orig
        bot.db.delete_config(bot.CONFIG_KEY_PROXY_URL)

    def _query(self):
        query = MagicMock()
        query.edit_message_text = AsyncMock()
        return query

    def _texts(self, query):
        return [c.kwargs["text"] for c in query.edit_message_text.await_args_list]

    def test_keyboard_has_both_test_buttons(self):
        callbacks = [b.callback_data for row in bot.build_koboldtest_keyboard().rows for b in row]
        self.assertTrue({"kobold:ping", "kobold:probe", "menu:start"} <= set(callbacks))

    def test_start_keyboard_shows_test_button_only_with_proxy(self):
        with_proxy = [b.callback_data for row in bot.build_start_keyboard(True).rows for b in row]
        without = [b.callback_data for row in bot.build_start_keyboard(False).rows for b in row]
        self.assertIn("menu:koboldtest", with_proxy)
        self.assertNotIn("menu:koboldtest", without)

    def test_non_admin_is_denied(self):
        bot.is_authorized_admin = lambda uid: False
        bot._kobold_ping_sync = MagicMock()
        query = self._query()
        asyncio.run(bot._handle_koboldtest_callback(query, 5, "kobold:ping"))
        self.assertIn("Admin", self._texts(query)[-1])
        bot._kobold_ping_sync.assert_not_called()

    def test_no_proxy_configured(self):
        bot.db.delete_config(bot.CONFIG_KEY_PROXY_URL)
        query = self._query()
        asyncio.run(bot._handle_koboldtest_callback(query, 1, "kobold:probe"))
        self.assertIn("/setproxy", self._texts(query)[-1])

    def test_ping_flow_shows_progress_then_result(self):
        bot._kobold_ping_sync = MagicMock(return_value={
            "ok": True, "model": "koboldcpp/Magidonia", "latency_ms": 120, "detail": "HTTP 200 via /api/v1/model"})
        query = self._query()
        asyncio.run(bot._handle_koboldtest_callback(query, 1, "kobold:ping"))
        texts = self._texts(query)
        self.assertIn("⏳", texts[0])
        self.assertIn("terhubung", texts[-1])
        self.assertIn("Magidonia", texts[-1])

    def test_probe_flow_uses_default_probe_and_checks_expect_word(self):
        bot._kobold_probe_sync = MagicMock(return_value={
            "ok": True, "text": "Ahoy, matey. ANCHOR", "latency_ms": 4000, "error": None})
        query = self._query()
        asyncio.run(bot._handle_koboldtest_callback(query, 1, "kobold:probe"))
        bot._kobold_probe_sync.assert_called_once()
        self.assertEqual(bot._kobold_probe_sync.call_args.args[1], bot.KOBOLD_PROBE_TEXT)
        self.assertIn("✅", self._texts(query)[-1])

    def test_failed_ping_shows_error(self):
        bot._kobold_ping_sync = MagicMock(return_value={
            "ok": False, "model": None, "latency_ms": None, "detail": "HTTP 530: tunnel mati"})
        query = self._query()
        asyncio.run(bot._handle_koboldtest_callback(query, 1, "kobold:ping"))
        self.assertIn("Gagal", self._texts(query)[-1])

    def _command_update(self, text):
        update = MagicMock()
        update.effective_user.id = 1
        update.effective_message.text = text
        status = MagicMock()
        status.edit_text = AsyncMock()
        update.effective_message.reply_text = AsyncMock(return_value=status)
        return update, status

    def test_command_without_args_opens_menu(self):
        update, _ = self._command_update("/koboldtest")
        asyncio.run(bot.koboldtest_command(update, MagicMock()))
        markup = update.effective_message.reply_text.await_args.kwargs["reply_markup"]
        self.assertIn("kobold:ping", [b.callback_data for row in markup.rows for b in row])

    def test_command_with_custom_scenario_runs_probe_without_expect_word(self):
        bot._kobold_probe_sync = MagicMock(return_value={
            "ok": True, "text": "Balasan bebas tanpa kata khusus", "latency_ms": 1000, "error": None})
        update, status = self._command_update("/koboldtest skenario saya sendiri")
        asyncio.run(bot.koboldtest_command(update, MagicMock()))
        self.assertEqual(bot._kobold_probe_sync.call_args.args[1], "skenario saya sendiri")
        self.assertIn("✅", status.edit_text.await_args.args[0])  # tidak ada kata wajib untuk pesan kustom

    def test_command_rejects_non_admin(self):
        bot.is_authorized_admin = lambda uid: False
        update, _ = self._command_update("/koboldtest")
        asyncio.run(bot.koboldtest_command(update, MagicMock()))
        self.assertIn("Admin", update.effective_message.reply_text.await_args.args[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
