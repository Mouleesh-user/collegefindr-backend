import os
import re
import time
import hashlib
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import bcrypt
import bleach
import jwt
from flask import Flask, g, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from pydantic import BaseModel, EmailStr, ValidationError
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in some deployments
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent

if load_dotenv:
    # Load local env files for development while keeping platform env vars as source of truth.
    for env_path in (BASE_DIR / ".env", BASE_DIR / ".enc"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


def _normalize_database_url(value: str) -> str:
    if value.startswith("postgres://"):
        return value.replace("postgres://", "postgresql://", 1)
    return value


def _normalize_origin(origin: str) -> str:
    return origin.strip().rstrip("/")


def _get_allowed_origins() -> List[str]:
    """Allow explicitly configured browser origins only (no wildcard)."""
    configured = os.getenv("ALLOWED_ORIGINS", "").strip()
    candidates: List[str] = []

    if configured:
        candidates.extend(item.strip() for item in configured.split(",") if item.strip())

    single_origin = os.getenv("ALLOWED_ORIGIN", "").strip()
    if single_origin:
        candidates.append(single_origin)

    if not candidates:
        candidates.append("https://mouleesh-user.github.io")

    if os.getenv("FLASK_DEBUG", "0") == "1":
        candidates.extend(["http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:5173"])

    normalized = [_normalize_origin(item) for item in candidates if item]
    # Preserve ordering while removing duplicates.
    return list(dict.fromkeys(normalized))


def _is_valid_origin_format(origin: str) -> bool:
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _model_field_names(model: Any) -> set[str]:
    if hasattr(model, "model_fields") and isinstance(model.model_fields, dict):
        return set(model.model_fields.keys())
    fields = getattr(model, "__fields__", {})
    if isinstance(fields, dict):
        return set(fields.keys())
    return set()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_model(model: Any, payload: Dict[str, Any]) -> Any:
    """Support both Pydantic v1 and v2 parsing APIs."""
    if hasattr(model, "model_validate"):
        return model.model_validate(payload)
    return model.parse_obj(payload)


def _serialize_model(instance: Any) -> Dict[str, Any]:
    if hasattr(instance, "model_dump"):
        return instance.model_dump(exclude_none=True)
    return instance.dict(exclude_none=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sanitize_text(value: str, max_length: int = 1000) -> str:
    cleaned = bleach.clean((value or "").strip(), tags=[], attributes={}, strip=True, strip_comments=True)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned[:max_length].strip()


def _sanitize_multiline_text(value: str, max_length: int = 5000) -> str:
    cleaned = bleach.clean((value or "").strip(), tags=[], attributes={}, strip=True, strip_comments=True)
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned[:max_length].strip()


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

database_url = _normalize_database_url(os.getenv("DATABASE_URL", "sqlite:///collegefindr.db"))
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["RATELIMIT_STORAGE_URI"] = os.getenv("RATELIMIT_STORAGE_URI", "memory://")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}
ALLOWED_ORIGINS = _get_allowed_origins()

for configured_origin in ALLOWED_ORIGINS:
    if not _is_valid_origin_format(configured_origin):
        raise RuntimeError(f"Invalid origin configured in ALLOWED_ORIGINS/ALLOWED_ORIGIN: {configured_origin}")

CORS(
    app,
    resources={
        r"/*": {"origins": ALLOWED_ORIGINS},
    },
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-CLIENT-KEY"],
    supports_credentials=False,
    max_age=600,
)

db = SQLAlchemy(app)
limiter = Limiter(key_func=get_remote_address, default_limits=["300 per day", "100 per hour"])
limiter.init_app(app)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-change-this-secret-with-at-least-32-characters")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
CLIENT_API_KEY = os.getenv("CLIENT_API_KEY", "").strip()
BOT_HEADER_NAME = "X-Requested-With"
BOT_HEADER_VALUE = os.getenv("BOT_HEADER_VALUE", "CollegeFindrWeb").strip()
CAPTCHA_ENABLED = os.getenv("CAPTCHA_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
HCAPTCHA_SECRET_KEY = os.getenv("HCAPTCHA_SECRET_KEY", "").strip()
HCAPTCHA_VERIFY_URL = "https://hcaptcha.com/siteverify"
CHAT_RATE_LIMIT = os.getenv("CHAT_RATE_LIMIT", "20 per minute")
CHAT_DAILY_RATE_LIMIT = os.getenv("CHAT_DAILY_RATE_LIMIT", "200 per day")
CHAT_IP_RATE_LIMIT = os.getenv("CHAT_IP_RATE_LIMIT", "10 per minute")
CHAT_MESSAGE_MAX_LENGTH = int(os.getenv("CHAT_MESSAGE_MAX_LENGTH", "500"))
CHAT_DUPLICATE_WINDOW_SECONDS = int(os.getenv("CHAT_DUPLICATE_WINDOW_SECONDS", "8"))
OPENROUTER_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "20"))
OPENROUTER_MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", "2"))
REQUIRE_ALLOWED_ORIGIN = os.getenv("REQUIRE_ALLOWED_ORIGIN", "1").strip().lower() in {"1", "true", "yes", "on"}
SIGNUP_ENABLED = os.getenv("SIGNUP_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
SIGNUP_ACCESS_CODE = os.getenv("SIGNUP_ACCESS_CODE", "").strip()
ALLOWED_USER_EMAILS = {
    _sanitize_text(item.lower(), 255)
    for item in os.getenv("ALLOWED_USER_EMAILS", "").split(",")
    if item.strip()
}
ALLOWED_CHAT_CONTEXTS = {
    "chat-messages",
    "saved-chat-messages",
    "applications-chat-messages",
    "financial-aid-chat-messages",
    "settings-chat-messages",
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
SYSTEM_PROMPT = (
    "You are CollegeFindr, an AI assistant that helps students discover colleges based "
    "on interests, budget, location, and goals. Keep responses practical and easy to read. "
    "Use clear Markdown structure (headings, short bullet lists, and tables when useful). "
    "Avoid decorative symbols and do not output raw HTML."
)

if JWT_SECRET_KEY == "dev-change-this-secret-with-at-least-32-characters" and os.getenv("FLASK_DEBUG", "0") != "1":
    raise RuntimeError("JWT_SECRET_KEY must be set in production")
if not CLIENT_API_KEY and os.getenv("FLASK_DEBUG", "0") != "1":
    raise RuntimeError("CLIENT_API_KEY must be set in production")

RECENT_CHAT_REQUESTS: Dict[str, Dict[str, Any]] = {}


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(30), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)

    settings = db.relationship("UserSettings", backref="user", uselist=False, cascade="all, delete-orphan")


class UserSettings(db.Model):
    __tablename__ = "user_settings"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False)
    notify_deadlines = db.Column(db.Boolean, nullable=False, default=True)
    notify_weekly_recommendations = db.Column(db.Boolean, nullable=False, default=True)
    notify_marketing = db.Column(db.Boolean, nullable=False, default=False)
    notify_scholarships = db.Column(db.Boolean, nullable=False, default=True)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)


class Message(db.Model):
    __tablename__ = "messages"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    context = db.Column(db.String(60), nullable=False, default="chat-messages", index=True)
    role = db.Column(db.String(16), nullable=False)  # user | assistant
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utc_now, index=True)


class ContactMessage(db.Model):
    __tablename__ = "contact_messages"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utc_now)


class Application(db.Model):
    __tablename__ = "applications"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    college_name = db.Column(db.String(180), nullable=False)
    status = db.Column(db.String(20), nullable=False, default="pending")
    deadline = db.Column(db.String(20), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now)


class ApiRequestLog(db.Model):
    __tablename__ = "api_request_logs"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, default=_utc_now, index=True)
    ip_address = db.Column(db.String(64), nullable=False, index=True)
    method = db.Column(db.String(10), nullable=False)
    endpoint = db.Column(db.String(160), nullable=False, index=True)
    status_code = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    tokens_consumed = db.Column(db.Integer, nullable=False, default=0)
    prompt_tokens = db.Column(db.Integer, nullable=False, default=0)
    completion_tokens = db.Column(db.Integer, nullable=False, default=0)
    latency_ms = db.Column(db.Integer, nullable=False, default=0)


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone: Optional[str] = None
    access_code: Optional[str] = None
    captcha_token: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    captcha_token: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    context: str = "chat-messages"


class ContactRequest(BaseModel):
    name: str
    email: EmailStr
    message: str
    captcha_token: Optional[str] = None


class SettingsUpdateRequest(BaseModel):
    full_name: str
    phone: Optional[str] = None
    notify_deadlines: bool
    notify_weekly_recommendations: bool
    notify_marketing: bool
    notify_scholarships: bool


class ApplicationCreateRequest(BaseModel):
    college_name: str
    status: str = "pending"
    deadline: Optional[str] = None
    notes: Optional[str] = None


def _build_openrouter_payload(user_message: str) -> Dict[str, Any]:
    """Build the message payload sent to OpenRouter."""
    return {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }


def _build_openrouter_payload_with_history(history_messages: List[Message], user_message: str) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history_messages:
        role = "assistant" if msg.role == "assistant" else "user"
        messages.append({"role": role, "content": msg.content})

    messages.append({"role": "user", "content": user_message})
    return {"model": OPENROUTER_MODEL, "messages": messages}


def _extract_message_content(choice: Dict[str, Any]) -> str:
    """Extract text from OpenRouter choice payloads that may be string or list based."""
    message = choice.get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)

    return str(content)


def _clean_reply_text(raw_reply: str) -> str:
    """Clean provider artifacts while preserving markdown structure for the frontend."""
    text = raw_reply or ""

    # Remove reasoning tags and special transport tokens that may leak into output.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^|]+\|>", "", text)

    # Normalize newline variants while preserving markdown and unicode symbols.
    text = re.sub(r"\r\n?", "\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    return text


def _create_jwt_token(user: User) -> str:
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _extract_current_user_from_token(token: str) -> Optional[User]:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = int(payload.get("sub", "0"))
    except (jwt.InvalidTokenError, ValueError, TypeError):
        return None

    return db.session.get(User, user_id)


def _get_optional_authenticated_user() -> Optional[User]:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return None
    return _extract_current_user_from_token(token)


def _reject_if_disallowed_origin() -> Optional[Any]:
    if not REQUIRE_ALLOWED_ORIGIN:
        return None

    origin = request.headers.get("Origin", "").strip()
    if not origin:
        # Non-browser clients may not send Origin. CORS is handled separately by Flask-CORS.
        return None

    normalized_origin = _normalize_origin(origin)
    if normalized_origin not in ALLOWED_ORIGINS:
        return jsonify({"error": f"Origin not allowed: {normalized_origin}"}), 403

    return None


def _is_email_allowed(email: str) -> bool:
    if not ALLOWED_USER_EMAILS:
        return True
    return email in ALLOWED_USER_EMAILS


def _chat_rate_limit_key() -> str:
    user = _get_optional_authenticated_user()
    if user:
        return f"user:{user.id}"
    return get_remote_address()


def _has_valid_bot_headers() -> bool:
    if not request.headers.get("User-Agent", "").strip():
        return False

    if BOT_HEADER_VALUE:
        header_value = request.headers.get(BOT_HEADER_NAME, "").strip()
        if header_value != BOT_HEADER_VALUE:
            return False

    return True


def _reject_if_invalid_client_key() -> Optional[Any]:
    supplied_key = request.headers.get("X-CLIENT-KEY", "").strip()
    if not supplied_key or not CLIENT_API_KEY or supplied_key != CLIENT_API_KEY:
        app.logger.warning("Denied /chat due to invalid X-CLIENT-KEY from ip=%s", _client_ip())
        return jsonify({"error": "Unauthorized client key"}), 403

    if not _has_valid_bot_headers():
        app.logger.warning("Denied /chat due to failed bot header check from ip=%s", _client_ip())
        return jsonify({"error": "Bot protection check failed"}), 403

    return None


def client_key_required(fn: Any) -> Any:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        client_key_error = _reject_if_invalid_client_key()
        if client_key_error:
            return client_key_error
        return fn(*args, **kwargs)

    return wrapper


def _is_rapid_duplicate_message(identifier: str, message: str) -> bool:
    now = _utc_now()
    fingerprint = hashlib.sha256(message.encode("utf-8")).hexdigest()
    previous = RECENT_CHAT_REQUESTS.get(identifier)

    if previous:
        previous_time = previous.get("time")
        previous_fingerprint = previous.get("fingerprint")
        if isinstance(previous_time, datetime) and previous_fingerprint == fingerprint:
            elapsed = (now - previous_time).total_seconds()
            if elapsed < CHAT_DUPLICATE_WINDOW_SECONDS:
                return True

    RECENT_CHAT_REQUESTS[identifier] = {"fingerprint": fingerprint, "time": now}

    if len(RECENT_CHAT_REQUESTS) > 10000:
        cutoff = now - timedelta(minutes=10)
        stale_keys = [key for key, value in RECENT_CHAT_REQUESTS.items() if value.get("time", now) < cutoff]
        for key in stale_keys:
            RECENT_CHAT_REQUESTS.pop(key, None)

    return False


def _client_ip() -> str:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()

    for header_name in ("CF-Connecting-IP", "X-Real-IP"):
        value = request.headers.get(header_name, "").strip()
        if value:
            return value

    return (request.remote_addr or "unknown").strip() or "unknown"


def _extract_captcha_token_from_request() -> str:
    header_token = request.headers.get("X-Captcha-Token", "")
    if header_token:
        return _sanitize_text(header_token, 4096)

    body = request.get_json(silent=True) or {}
    if isinstance(body, dict):
        return _sanitize_text(str(body.get("captcha_token") or ""), 4096)
    return ""


def _verify_captcha(token: str) -> bool:
    if not CAPTCHA_ENABLED:
        return True

    if not HCAPTCHA_SECRET_KEY:
        app.logger.error("CAPTCHA_ENABLED is true but HCAPTCHA_SECRET_KEY is missing")
        return False

    if not token:
        return False

    requests_module = _get_requests_module()

    try:
        response = requests_module.post(
            HCAPTCHA_VERIFY_URL,
            data={
                "secret": HCAPTCHA_SECRET_KEY,
                "response": token,
                "remoteip": _client_ip(),
            },
            timeout=10,
        )
        if not response.ok:
            app.logger.warning("Captcha verification failed with status %s", response.status_code)
            return False
        data = response.json()
        return bool(data.get("success"))
    except Exception:
        app.logger.exception("Captcha verification request failed")
        return False


def captcha_required(fn: Any) -> Any:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not CAPTCHA_ENABLED:
            return fn(*args, **kwargs)

        token = _extract_captcha_token_from_request()
        if not token:
            return jsonify({"error": "Captcha token is required"}), 400

        if not _verify_captcha(token):
            return jsonify({"error": "Captcha verification failed"}), 403

        return fn(*args, **kwargs)

    return wrapper


def auth_required(fn: Any) -> Any:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        origin_error = _reject_if_disallowed_origin()
        if origin_error:
            return origin_error

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            app.logger.warning("Unauthorized request missing bearer token from ip=%s endpoint=%s", _client_ip(), request.path)
            return jsonify({"error": "Authorization token is required"}), 401

        token = auth_header.split(" ", 1)[1].strip()
        user = _extract_current_user_from_token(token)
        if not user:
            app.logger.warning("Unauthorized request with invalid token from ip=%s endpoint=%s", _client_ip(), request.path)
            return jsonify({"error": "Invalid or expired token"}), 401

        g.current_user = user
        return fn(*args, **kwargs)

    return wrapper


def _validate_json_request(model: Any) -> Any:
    if not request.is_json:
        return None, (jsonify({"error": "Request must be JSON"}), 400)

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return None, (jsonify({"error": "Request JSON must be an object"}), 400)

    unknown_fields = sorted(set(body.keys()) - _model_field_names(model))
    if unknown_fields:
        return None, (
            jsonify(
                {
                    "error": "Invalid request payload",
                    "details": [
                        {
                            "msg": f"Unexpected fields: {', '.join(unknown_fields)}",
                            "type": "value_error.extra",
                        }
                    ],
                }
            ),
            400,
        )

    try:
        parsed = _parse_model(model, body)
        return parsed, None
    except ValidationError as exc:
        return None, (jsonify({"error": "Invalid request payload", "details": exc.errors()}), 400)


def _extract_openrouter_usage(data: Dict[str, Any]) -> Dict[str, int]:
    usage = data.get("usage") or {}
    prompt_tokens = _safe_int(usage.get("prompt_tokens"), 0)
    completion_tokens = _safe_int(usage.get("completion_tokens"), 0)
    total_tokens = _safe_int(usage.get("total_tokens"), prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": max(prompt_tokens, 0),
        "completion_tokens": max(completion_tokens, 0),
        "total_tokens": max(total_tokens, 0),
    }


def _serialize_user(user: User) -> Dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "phone": user.phone,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _serialize_settings(settings: UserSettings) -> Dict[str, Any]:
    return {
        "notify_deadlines": settings.notify_deadlines,
        "notify_weekly_recommendations": settings.notify_weekly_recommendations,
        "notify_marketing": settings.notify_marketing,
        "notify_scholarships": settings.notify_scholarships,
        "updated_at": settings.updated_at.isoformat() if settings.updated_at else None,
    }


def _ensure_user_settings(user: User) -> UserSettings:
    settings = user.settings
    if settings:
        return settings

    settings = UserSettings(user_id=user.id)
    db.session.add(settings)
    db.session.commit()
    return settings


def _get_requests_module() -> Any:
    import requests as requests_module

    return requests_module


def _get_openrouter_reply(user_message: str) -> str:
    """
    API flow:
    1) Read API key from environment (never from frontend).
    2) Send the user message to OpenRouter.
    3) Extract and return assistant text.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = _build_openrouter_payload(user_message)

    requests_module = _get_requests_module()

    response = requests_module.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

    data = response.json()
    choices = data.get("choices", [])

    if not choices:
        raise RuntimeError("OpenRouter response did not include choices")

    raw_reply = _extract_message_content(choices[0]).strip()
    reply = _clean_reply_text(raw_reply)
    if not reply:
        raise RuntimeError("OpenRouter returned an empty response")

    return reply


def _get_openrouter_reply_with_history(user_message: str, history_messages: List[Message]) -> Dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = _build_openrouter_payload_with_history(history_messages, user_message)

    requests_module = _get_requests_module()

    last_error: Optional[Exception] = None

    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        try:
            response = requests_module.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=OPENROUTER_TIMEOUT_SECONDS,
            )

            retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
            if not response.ok:
                if response.status_code in retryable_statuses and attempt < OPENROUTER_MAX_RETRIES:
                    time.sleep(0.35 * (attempt + 1))
                    continue

                app.logger.error("OpenRouter API error status=%s body=%s", response.status_code, response.text[:500])
                raise RuntimeError("OpenRouter request failed")

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("OpenRouter response did not include choices")

            raw_reply = _extract_message_content(choices[0]).strip()
            reply = _clean_reply_text(raw_reply)
            if not reply:
                raise RuntimeError("OpenRouter returned an empty response")

            return {"reply": reply, "usage": _extract_openrouter_usage(data)}
        except requests_module.RequestException as exc:
            last_error = exc
            if attempt < OPENROUTER_MAX_RETRIES:
                time.sleep(0.35 * (attempt + 1))
                continue

    if last_error:
        raise RuntimeError("Failed to reach AI provider") from last_error

    raise RuntimeError("OpenRouter request failed")


@app.before_request
def start_request_context() -> None:
    g.request_started_at = _utc_now()
    g.tokens_consumed = 0
    g.prompt_tokens = 0
    g.completion_tokens = 0
    content_length = request.content_length
    if content_length is None:
        try:
            content_length = len(request.get_data(cache=True) or b"")
        except Exception:
            content_length = 0
    g.request_size_bytes = max(_safe_int(content_length, 0), 0)


@app.after_request
def audit_request(response: Any) -> Any:
    started_at = getattr(g, "request_started_at", _utc_now())
    elapsed_ms = int(max(((_utc_now() - started_at).total_seconds() * 1000), 0))
    current_user = getattr(g, "current_user", None)

    log_row = ApiRequestLog(
        timestamp=started_at,
        ip_address=_client_ip(),
        method=request.method,
        endpoint=request.path,
        status_code=int(response.status_code),
        user_id=current_user.id if current_user else None,
        tokens_consumed=_safe_int(getattr(g, "tokens_consumed", 0), 0),
        prompt_tokens=_safe_int(getattr(g, "prompt_tokens", 0), 0),
        completion_tokens=_safe_int(getattr(g, "completion_tokens", 0), 0),
        latency_ms=elapsed_ms,
    )

    try:
        db.session.add(log_row)
        db.session.commit()
    except Exception:
        db.session.rollback()
        app.logger.exception("Failed to write ApiRequestLog")

    app.logger.info(
        "api_request ip=%s time=%s endpoint=%s method=%s status=%s bytes=%s tokens=%s prompt=%s completion=%s latency_ms=%s",
        log_row.ip_address,
        log_row.timestamp.isoformat(),
        log_row.endpoint,
        log_row.method,
        log_row.status_code,
        _safe_int(getattr(g, "request_size_bytes", 0), 0),
        log_row.tokens_consumed,
        log_row.prompt_tokens,
        log_row.completion_tokens,
        log_row.latency_ms,
    )

    if request.path.rstrip("/") == "/chat":
        app.logger.info(
            "chat_call ip=%s timestamp=%s endpoint=%s bytes=%s status=%s",
            log_row.ip_address,
            log_row.timestamp.isoformat(),
            log_row.endpoint,
            _safe_int(getattr(g, "request_size_bytes", 0), 0),
            log_row.status_code,
        )

    return response


@app.get("/")
def health() -> Any:
    return (
        jsonify(
            {
                "status": "ok",
                "service": "CollegeFindr Flask API",
                "chat_endpoint": "POST /chat",
                "auth_endpoints": ["POST /auth/register", "POST /auth/login", "GET /auth/me"],
            }
        ),
        200,
    )


@app.get("/health")
def healthcheck() -> Any:
    return jsonify({"status": "ok", "service": "CollegeFindr Flask API"}), 200


@app.route("/ping", methods=["GET", "OPTIONS"])
def ping() -> Any:
    if request.method == "OPTIONS":
        return ("", 204)
    return jsonify({"status": "ok", "timestamp": _utc_now().isoformat()}), 200


@app.route("/<path:_path>", methods=["OPTIONS"])
def options_fallback(_path: str) -> Any:
    return ("", 204)


@app.route("/", methods=["OPTIONS"])
def options_root() -> Any:
    return ("", 204)


@app.post("/auth/register")
@limiter.limit("10 per minute")
@captcha_required
def register() -> Any:
    origin_error = _reject_if_disallowed_origin()
    if origin_error:
        return origin_error

    if not SIGNUP_ENABLED:
        return jsonify({"error": "Signups are currently disabled"}), 403

    parsed, error_response = _validate_json_request(RegisterRequest)
    if error_response:
        return error_response

    payload = _serialize_model(parsed)
    email = _sanitize_text(payload["email"].lower(), 255)
    password = payload["password"]
    full_name = _sanitize_text(payload["full_name"], 120)
    phone = _sanitize_text(payload.get("phone") or "", 30) or None
    access_code = _sanitize_text(payload.get("access_code") or "", 128)

    if not _is_email_allowed(email):
        return jsonify({"error": "This email is not allowed to register"}), 403

    if SIGNUP_ACCESS_CODE and access_code != SIGNUP_ACCESS_CODE:
        return jsonify({"error": "Invalid signup access code"}), 403

    if len(password) < 8 or len(password) > 128:
        return jsonify({"error": "Password must be between 8 and 128 characters"}), 400
    if len(full_name) < 2:
        return jsonify({"error": "Full name must be at least 2 characters"}), 400
    if phone and len(phone) < 6:
        return jsonify({"error": "Phone must be at least 6 characters"}), 400

    existing = User.query.filter_by(email=email).first()
    if existing:
        return jsonify({"error": "An account with this email already exists"}), 409

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    user = User(email=email, password_hash=password_hash, full_name=full_name, phone=phone)
    db.session.add(user)
    db.session.flush()
    settings = UserSettings(user_id=user.id)
    db.session.add(settings)
    db.session.commit()

    token = _create_jwt_token(user)
    return jsonify({"token": token, "user": _serialize_user(user), "settings": _serialize_settings(settings)}), 201


@app.post("/auth/login")
@limiter.limit("20 per minute")
@captcha_required
def login() -> Any:
    origin_error = _reject_if_disallowed_origin()
    if origin_error:
        return origin_error

    parsed, error_response = _validate_json_request(LoginRequest)
    if error_response:
        return error_response

    payload = _serialize_model(parsed)
    email = _sanitize_text(payload["email"].lower(), 255)
    password = payload["password"]

    if not _is_email_allowed(email):
        return jsonify({"error": "This email is not allowed to log in"}), 403

    if len(password) < 8 or len(password) > 128:
        return jsonify({"error": "Password must be between 8 and 128 characters"}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "Invalid email or password"}), 401

    if not bcrypt.checkpw(password.encode("utf-8"), user.password_hash.encode("utf-8")):
        return jsonify({"error": "Invalid email or password"}), 401

    settings = _ensure_user_settings(user)
    token = _create_jwt_token(user)
    return jsonify({"token": token, "user": _serialize_user(user), "settings": _serialize_settings(settings)}), 200


@app.get("/auth/me")
@auth_required
@limiter.limit("60 per minute")
def auth_me() -> Any:
    user: User = g.current_user
    settings = _ensure_user_settings(user)
    return jsonify({"user": _serialize_user(user), "settings": _serialize_settings(settings)}), 200


@app.get("/chat")
@app.get("/chat/")
def chat_help() -> Any:
    return (
        jsonify(
            {
                "message": "Use POST /chat with JSON body: {\"message\": \"your text\"}",
            }
        ),
        200,
    )


@app.post("/chat")
@app.post("/chat/")
@client_key_required
@auth_required
@limiter.limit(CHAT_IP_RATE_LIMIT, key_func=get_remote_address)
def chat() -> Any:
    parsed, error_response = _validate_json_request(ChatRequest)
    if error_response:
        return error_response

    payload = _serialize_model(parsed)
    context = _sanitize_text(payload.get("context") or "chat-messages", 60)
    if context not in ALLOWED_CHAT_CONTEXTS:
        return jsonify({"error": "Invalid chat context"}), 400

    raw_message = str(payload.get("message") or "")
    if len(raw_message.strip()) > CHAT_MESSAGE_MAX_LENGTH:
        return jsonify({"error": f"message exceeds {CHAT_MESSAGE_MAX_LENGTH} characters"}), 400

    message = _sanitize_multiline_text(raw_message, CHAT_MESSAGE_MAX_LENGTH)
    if not message:
        return jsonify({"error": "message is required"}), 400

    user: User = g.current_user
    duplicate_key = f"user:{user.id}:{_client_ip()}"
    if _is_rapid_duplicate_message(duplicate_key, message):
        return jsonify({"error": "Repeated requests detected. Please wait before retrying."}), 429

    requests_module = _get_requests_module()

    try:
        history = (
            Message.query.filter_by(user_id=user.id, context=context)
            .order_by(Message.created_at.desc())
            .limit(8)
            .all()
        )
        history.reverse()
        llm_result = _get_openrouter_reply_with_history(message, history)
        reply = str(llm_result.get("reply") or "")
        usage = llm_result.get("usage") or {}

        reply = _sanitize_multiline_text(reply, 5000)
        g.prompt_tokens = _safe_int(usage.get("prompt_tokens"), 0)
        g.completion_tokens = _safe_int(usage.get("completion_tokens"), 0)
        g.tokens_consumed = _safe_int(usage.get("total_tokens"), g.prompt_tokens + g.completion_tokens)

        response_messages = [
            {
                "id": None,
                "role": "user",
                "content": message,
                "created_at": None,
            },
            {
                "id": None,
                "role": "assistant",
                "content": reply,
                "created_at": None,
            },
        ]

        user_msg = Message(user_id=user.id, context=context, role="user", content=message)
        assistant_msg = Message(user_id=user.id, context=context, role="assistant", content=reply)
        db.session.add(user_msg)
        db.session.add(assistant_msg)
        db.session.commit()

        response_messages = [
            {
                "id": user_msg.id,
                "role": user_msg.role,
                "content": user_msg.content,
                "created_at": user_msg.created_at.isoformat() if user_msg.created_at else None,
            },
            {
                "id": assistant_msg.id,
                "role": assistant_msg.role,
                "content": assistant_msg.content,
                "created_at": assistant_msg.created_at.isoformat() if assistant_msg.created_at else None,
            },
        ]

        return (
            jsonify(
                {
                    "reply": reply,
                    "context": context,
                    "messages": response_messages,
                    "usage": {
                        "prompt_tokens": g.prompt_tokens,
                        "completion_tokens": g.completion_tokens,
                        "total_tokens": g.tokens_consumed,
                    },
                }
            ),
            200,
        )
    except ValueError as exc:
        message = str(exc)
        if "OPENROUTER_API_KEY" in message:
            return jsonify({"error": "AI service is not configured on the server"}), 503
        return jsonify({"error": message}), 400
    except requests_module.RequestException:
        return jsonify({"error": "Failed to reach AI provider"}), 502
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception:
        return jsonify({"error": "Unexpected server error"}), 500


@app.get("/messages/<string:context>")
@auth_required
@limiter.limit("60 per minute")
def get_messages(context: str) -> Any:
    safe_context = _sanitize_text(context, 60)
    if safe_context not in ALLOWED_CHAT_CONTEXTS:
        return jsonify({"error": "Invalid chat context"}), 400

    user: User = g.current_user
    messages = (
        Message.query.filter_by(user_id=user.id, context=safe_context)
        .order_by(Message.created_at.asc())
        .limit(200)
        .all()
    )

    return (
        jsonify(
            {
                "context": safe_context,
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    }
                    for msg in messages
                ],
            }
        ),
        200,
    )


@app.post("/contact")
@limiter.limit("10 per hour")
@captcha_required
def create_contact_message() -> Any:
    origin_error = _reject_if_disallowed_origin()
    if origin_error:
        return origin_error

    parsed, error_response = _validate_json_request(ContactRequest)
    if error_response:
        return error_response

    payload = _serialize_model(parsed)
    user = _get_optional_authenticated_user()

    contact = ContactMessage(
        user_id=user.id if user else None,
        name=_sanitize_text(payload["name"], 120),
        email=_sanitize_text(payload["email"].lower(), 255),
        message=_sanitize_multiline_text(payload["message"], 3000),
    )

    if len(contact.name) < 2:
        return jsonify({"error": "Name must be at least 2 characters"}), 400
    if len(contact.message) < 2:
        return jsonify({"error": "Message must be at least 2 characters"}), 400

    db.session.add(contact)
    db.session.commit()
    return jsonify({"message": "Your message has been sent successfully"}), 201


@app.get("/settings")
@auth_required
@limiter.limit("60 per minute")
def get_settings() -> Any:
    user: User = g.current_user
    settings = _ensure_user_settings(user)
    return jsonify({"user": _serialize_user(user), "settings": _serialize_settings(settings)}), 200


@app.put("/settings")
@auth_required
@limiter.limit("20 per minute")
def update_settings() -> Any:
    parsed, error_response = _validate_json_request(SettingsUpdateRequest)
    if error_response:
        return error_response

    payload = _serialize_model(parsed)
    user: User = g.current_user
    settings = _ensure_user_settings(user)

    user.full_name = _sanitize_text(payload["full_name"], 120)
    user.phone = _sanitize_text(payload.get("phone") or "", 30) or None

    if len(user.full_name) < 2:
        return jsonify({"error": "Full name must be at least 2 characters"}), 400
    if user.phone and len(user.phone) < 6:
        return jsonify({"error": "Phone must be at least 6 characters"}), 400

    settings.notify_deadlines = bool(payload["notify_deadlines"])
    settings.notify_weekly_recommendations = bool(payload["notify_weekly_recommendations"])
    settings.notify_marketing = bool(payload["notify_marketing"])
    settings.notify_scholarships = bool(payload["notify_scholarships"])
    db.session.commit()

    return jsonify({"message": "Settings updated", "user": _serialize_user(user), "settings": _serialize_settings(settings)}), 200


@app.get("/applications")
@auth_required
@limiter.limit("60 per minute")
def list_applications() -> Any:
    user: User = g.current_user
    applications = (
        Application.query.filter_by(user_id=user.id)
        .order_by(Application.created_at.desc())
        .all()
    )
    return (
        jsonify(
            {
                "applications": [
                    {
                        "id": app_item.id,
                        "college_name": app_item.college_name,
                        "status": app_item.status,
                        "deadline": app_item.deadline,
                        "notes": app_item.notes,
                        "created_at": app_item.created_at.isoformat() if app_item.created_at else None,
                        "updated_at": app_item.updated_at.isoformat() if app_item.updated_at else None,
                    }
                    for app_item in applications
                ]
            }
        ),
        200,
    )


@app.post("/applications")
@auth_required
@limiter.limit("20 per minute")
def create_application() -> Any:
    parsed, error_response = _validate_json_request(ApplicationCreateRequest)
    if error_response:
        return error_response

    payload = _serialize_model(parsed)
    status = _sanitize_text(payload.get("status") or "pending", 20).lower()
    if status not in {"pending", "submitted", "accepted", "rejected"}:
        return jsonify({"error": "Invalid status"}), 400

    user: User = g.current_user
    college_name = _sanitize_text(payload["college_name"], 180)
    if len(college_name) < 2:
        return jsonify({"error": "College name must be at least 2 characters"}), 400

    notes = _sanitize_multiline_text(payload.get("notes") or "", 1500) or None

    application = Application(
        user_id=user.id,
        college_name=college_name,
        status=status,
        deadline=_sanitize_text(payload.get("deadline") or "", 20) or None,
        notes=notes,
    )
    db.session.add(application)
    db.session.commit()

    return (
        jsonify(
            {
                "message": "Application saved",
                "application": {
                    "id": application.id,
                    "college_name": application.college_name,
                    "status": application.status,
                    "deadline": application.deadline,
                    "notes": application.notes,
                    "created_at": application.created_at.isoformat() if application.created_at else None,
                    "updated_at": application.updated_at.isoformat() if application.updated_at else None,
                },
            }
        ),
        201,
    )


@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException) -> Any:
    message = exc.description if getattr(exc, "description", None) else "Request failed"
    return jsonify({"error": str(message)}), int(exc.code or 500)


@app.errorhandler(Exception)
def handle_unexpected_exception(exc: Exception) -> Any:
    app.logger.exception("Unhandled server exception: %s", exc)
    return jsonify({"error": "Unexpected server error"}), 500


def _should_auto_create_tables() -> bool:
    toggle = os.getenv("AUTO_CREATE_DB", "").strip().lower()
    if toggle in {"1", "true", "yes", "on"}:
        return True
    if toggle in {"0", "false", "no", "off"}:
        return False
    return database_url.startswith("sqlite:///")


if _should_auto_create_tables():
    with app.app_context():
        db.create_all()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
