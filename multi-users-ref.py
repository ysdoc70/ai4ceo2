import hashlib
import os
import secrets
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parent.parent
load_dotenv(_APP_DIR / ".env")
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

PBKDF2_ITERS = 390_000


def hash_password(password: str, salt_hex: Optional[str] = None) -> Tuple[str, str]:
    if salt_hex is None:
        salt_bytes = secrets.token_bytes(16)
        salt_hex = salt_bytes.hex()
    else:
        salt_bytes = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERS)
    return dk.hex(), salt_hex


def verify_password(password: str, salt_hex: str, stored_hash_hex: str) -> bool:
    salt_bytes = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERS)
    return secrets.compare_digest(dk.hex(), stored_hash_hex)


def normalize_username(raw: str) -> str:
    u = raw.strip()
    if not u or len(u) > 128:
        raise ValueError("사용자 ID는 1~128자여야 합니다.")
    return u


def _sync_streamlit_secrets_to_env() -> None:
    """secrets.toml 이 없으면 secrets 접근 시 예외가 날 수 있어, 단계별로 잡는다."""
    try:
        sec = st.secrets
    except Exception:
        return
    for key in (
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_ANON_KEY",
        "SUPABASE_PUBLISHABLE_KEY",
    ):
        if key in os.environ:
            continue
        try:
            if key in sec:
                os.environ[key] = str(sec[key])
        except Exception:
            return


def _is_missing_db_schema_error(exc: BaseException) -> bool:
    raw = str(exc).lower()
    if "user_id" in raw and ("does not exist" in raw or "could not find" in raw):
        return True
    if "app_users" in raw and ("does not exist" in raw or "schema cache" in raw):
        return True
    if "match_rag_documents" in raw and (
        "could not find" in raw
        or "function" in raw
        or "argument" in raw
        or "not match" in raw
        or "pgrst202" in raw
    ):
        return True
    if "42703" in raw:
        return True
    return False


def _schema_error_stop(exc: BaseException) -> None:
    raw = str(exc).lower()
    quick = ""
    if "app_users" in raw and ("pgrst205" in raw or "schema cache" in raw or "could not find the table" in raw):
        quick = (
            "\n\n**지금 오류가 `app_users` 테이블 없음이면:** "
            "`prompt/10.multi-users/fix_app_users_only.sql` 내용을 SQL Editor에 붙여 실행한 뒤, "
            "브라우저에서 앱을 새로고침하세요.\n\n"
            "그다음에도 `user_id` 컬럼·`match_rag_documents` 관련 오류가 나오면 전체 스키마 스크립트를 실행해야 합니다.\n\n"
        )
    st.error(
        "DB 스키마가 맞지 않습니다. Supabase SQL Editor에서 순서대로 실행하세요:\n\n"
        "1. `prompt/10.multi-session/multi-session-ref.sql`\n"
        "2. `prompt/10.multi-users/create_app_users_schema.sql` (동일: `multi-users-app-auth.sql`)\n\n"
        f"{quick}"
        f"(원본 오류: {exc})"
    )
    st.stop()


_sync_streamlit_secrets_to_env()

st.set_page_config(page_title="PDF 기반 멀티유저 RAG 챗봇", page_icon="📚", layout="wide")

st.markdown(
    """
<style>
h1 { font-size: 1.4rem !important; font-weight: 600 !important; color: #ff69b4 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #ffd700 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1f77b4 !important; }
.stChatMessage { font-size: 0.95rem !important; line-height: 1.5 !important; }
.stChatMessage * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; }
.stButton > button { background-color: #ff69b4 !important; color: white !important; border: none !important;
  border-radius: 5px !important; font-weight: bold !important; }
.stButton > button:hover { background-color: #ff1493 !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="text-align: center; margin-top: -3rem; margin-bottom: 0.5rem;">
  <h1 style="font-size: 2.3rem; font-weight: bold; margin: 0;">
    <span style="color: #1f77b4;">PDF</span>
    <span style="color: #9b59b6;">기반 멀티유저</span>
    <span style="color: #ffd700;">RAG</span>
    <span style="color: #d62728;">챗봇</span>
  </h1>
</div>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "chat_history": [],
        "current_session_id": None,
        "selected_session_id": None,
        "selected_model": "gpt-5.5",
        "processed_files": [],
        "active_sources": [],
        "bootstrapped": False,
        "retriever_ready": False,
        "user_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def _supabase_db_key() -> Optional[str]:
    sr = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if sr:
        return sr
    anon = os.getenv("SUPABASE_ANON_KEY", "").strip()
    pub = os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
    if anon.startswith("eyJ"):
        return anon
    if pub.startswith("eyJ"):
        return pub
    return anon or pub


def _supabase_key_warning() -> Optional[str]:
    if os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        return None
    return (
        "`app_users` 테이블을 쓰려면 **`SUPABASE_SERVICE_ROLE_KEY`** 를 Secrets 또는 .env 에 넣는 것을 권장합니다. "
        "anon 키만 쓰면 RLS 때문에 가입·로그인이 막힐 수 있습니다."
    )


def make_supabase() -> Optional[Client]:
    url = os.getenv("SUPABASE_URL")
    key = _supabase_db_key()
    if not url or not key:
        return None
    return create_client(url, key)


def ensure_api_keys(openai_key: str, anthropic_key: str, gemini_key: str) -> None:
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key.strip()
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key.strip()
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key.strip()


def sign_in(client: Client, username: str, password: str) -> bool:
    try:
        u = normalize_username(username)
    except ValueError as e:
        st.error(str(e))
        return False
    try:
        res = client.table("app_users").select("password_hash,salt").eq("username", u).limit(1).execute()
        rows = res.data or []
        if not rows:
            st.error("사용자 ID 또는 비밀번호가 올바르지 않습니다.")
            return False
        row = rows[0]
        if verify_password(password, row["salt"], row["password_hash"]):
            st.session_state.user_id = u
            st.session_state.bootstrapped = False
            return True
        st.error("사용자 ID 또는 비밀번호가 올바르지 않습니다.")
        return False
    except Exception as e:
        if _is_missing_db_schema_error(e):
            _schema_error_stop(e)
        st.error(f"로그인 오류: {e}")
        return False


def sign_out() -> None:
    st.session_state.user_id = None
    st.session_state.bootstrapped = False
    clear_screen()


def get_embeddings() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API 키가 필요합니다. 사이드바에서 입력하세요.")
    return OpenAIEmbeddings(openai_api_key=api_key)


def build_llm(model_name: str) -> Any:
    if model_name == "gpt-5.5":
        return ChatOpenAI(model="gpt-5.5", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
    if model_name == "claude-opus-4-7":
        return ChatAnthropic(model="claude-opus-4-7", temperature=0.7, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
    if model_name == "gemini-3-pro-preview":
        return ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview", temperature=0.7, google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    raise ValueError(f"지원하지 않는 모델: {model_name}")


def fetch_sessions(supabase: Client, user_id: str) -> List[Dict[str, Any]]:
    try:
        res = (
            supabase.table("rag_sessions")
            .select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(200)
            .execute()
        )
        return res.data or []
    except Exception as e:
        if _is_missing_db_schema_error(e):
            _schema_error_stop(e)
        raise


def generate_session_title() -> str:
    user_msg = next((m["content"] for m in st.session_state.chat_history if m["role"] == "user"), "").strip()
    ai_msg = next((m["content"] for m in st.session_state.chat_history if m["role"] == "assistant"), "").strip()
    if not user_msg:
        return "새 세션"
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return user_msg[:30]
        llm = ChatOpenAI(model="gpt-5.5", temperature=0.3, openai_api_key=api_key)
        prompt = f"질문과 답변 핵심으로 한국어 세션 제목 18자 이내 1개만 출력.\n질문:{user_msg[:500]}\n답변:{ai_msg[:700]}"
        title = str(llm.invoke(prompt).content).strip().replace("\n", " ")
        return title[:30] if title else user_msg[:30]
    except Exception:
        return user_msg[:30]


def upsert_session(supabase: Client, user_id: str) -> bool:
    if not st.session_state.current_session_id:
        st.session_state.current_session_id = str(uuid.uuid4())
    title = generate_session_title()
    payload = {
        "id": st.session_state.current_session_id,
        "title": title,
        "model_name": st.session_state.selected_model,
        "user_id": user_id,
    }
    supabase.table("rag_sessions").upsert(payload).execute()
    supabase.table("rag_messages").delete().eq("session_id", st.session_state.current_session_id).execute()
    rows = []
    for i, msg in enumerate(st.session_state.chat_history):
        rows.append(
            {
                "session_id": st.session_state.current_session_id,
                "turn_index": i,
                "role": msg["role"],
                "content": str(msg["content"]),
            }
        )
    if rows:
        supabase.table("rag_messages").insert(rows).execute()
    return True


def ensure_session_row(supabase: Client, session_id: str, model_name: str, user_id: str) -> None:
    if not session_id:
        raise ValueError("session_id가 비어 있습니다.")
    supabase.table("rag_sessions").upsert(
        {
            "id": session_id,
            "title": "새 세션",
            "model_name": model_name,
            "user_id": user_id,
        }
    ).execute()


def load_session(supabase: Client, session_id: str, user_id: str) -> bool:
    own = (
        supabase.table("rag_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not (own.data or []):
        st.error("해당 세션을 불러올 권한이 없거나 존재하지 않습니다.")
        return False
    res = (
        supabase.table("rag_messages")
        .select("role,content,turn_index")
        .eq("session_id", session_id)
        .order("turn_index")
        .execute()
    )
    st.session_state.chat_history = [{"role": r["role"], "content": r["content"]} for r in (res.data or [])]
    st.session_state.current_session_id = session_id
    doc_rows = (
        supabase.table("rag_session_documents")
        .select("source_name")
        .eq("session_id", session_id)
        .order("source_name")
        .execute()
    )
    st.session_state.active_sources = sorted({r["source_name"] for r in (doc_rows.data or [])})
    st.session_state.retriever_ready = bool(st.session_state.active_sources)
    st.session_state.processed_files = list(st.session_state.active_sources)
    return True


def clear_screen() -> None:
    st.session_state.chat_history = []
    st.session_state.processed_files = []
    st.session_state.active_sources = []
    st.session_state.retriever_ready = False


def delete_session(supabase: Client, session_id: str, user_id: str) -> bool:
    supabase.table("rag_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()
    if st.session_state.current_session_id == session_id:
        st.session_state.current_session_id = None
        clear_screen()
    return True


def save_chunks_to_vector_db(
    supabase: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    chunks: List[Document],
    source_name: str,
    user_id: str,
) -> int:
    unique_chunks: List[Document] = []
    for ch in chunks:
        text = str(ch.page_content).strip()
        if not text:
            continue
        hash_value = hashlib.sha256(text.encode("utf-8")).hexdigest()
        exists = (
            supabase.table("rag_documents")
            .select("id")
            .eq("content_hash", hash_value)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if exists.data:
            link_exists = (
                supabase.table("rag_session_documents")
                .select("id")
                .eq("session_id", session_id)
                .eq("document_id", exists.data[0]["id"])
                .limit(1)
                .execute()
            )
            if not link_exists.data:
                supabase.table("rag_session_documents").insert(
                    {"session_id": session_id, "document_id": exists.data[0]["id"], "source_name": source_name}
                ).execute()
            continue
        ch.metadata = ch.metadata or {}
        ch.metadata["content_hash"] = hash_value
        unique_chunks.append(ch)

    if not unique_chunks:
        return 0

    vectors = embeddings.embed_documents([u.page_content for u in unique_chunks])
    inserted = 0
    for ch, vec in zip(unique_chunks, vectors):
        record = (
            supabase.table("rag_documents")
            .insert(
                {
                    "content": ch.page_content,
                    "metadata": ch.metadata,
                    "content_hash": ch.metadata["content_hash"],
                    "embedding": vec,
                    "user_id": user_id,
                }
            )
            .execute()
        )
        if record.data:
            inserted += 1
            supabase.table("rag_session_documents").insert(
                {"session_id": session_id, "document_id": record.data[0]["id"], "source_name": source_name}
            ).execute()
    return inserted


def retrieve_docs(
    supabase: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    app_user_id: str,
    query: str,
    k: int = 5,
) -> List[Document]:
    query_vec = embeddings.embed_query(query)
    try:
        res = supabase.rpc(
            "match_rag_documents",
            {
                "query_embedding": query_vec,
                "target_session_id": session_id,
                "match_app_user_id": app_user_id,
                "match_count": k,
                "match_threshold": 0.6,
            },
        ).execute()
    except Exception as e:
        if _is_missing_db_schema_error(e):
            _schema_error_stop(e)
        raise
    docs: List[Document] = []
    for row in res.data or []:
        docs.append(Document(page_content=row["content"], metadata=row.get("metadata", {}) or {}))
    return docs


def generate_followups(question: str, answer: str) -> List[str]:
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return []
        llm = ChatOpenAI(model="gpt-5.5", temperature=0.8, openai_api_key=api_key)
        prompt = f"질문:{question}\n답변:{answer[:1200]}\n다음에 물어볼 질문 3개만 줄바꿈으로 출력."
        text = str(llm.invoke(prompt).content).strip()
        items = [x.strip("-• 1234567890. ").strip() for x in text.split("\n") if x.strip()]
        return items[:3]
    except Exception:
        return []


def maybe_bootstrap(supabase: Client, uid: Optional[str]) -> None:
    if st.session_state.bootstrapped:
        return
    if not uid:
        st.session_state.current_session_id = None
        clear_screen()
        st.session_state.bootstrapped = True
        return
    sessions = fetch_sessions(supabase, uid)
    if sessions:
        load_session(supabase, sessions[0]["id"], uid)
        st.session_state.selected_session_id = sessions[0]["id"]
    else:
        st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.bootstrapped = True


supabase = make_supabase()
if not supabase:
    st.error("SUPABASE_URL 및 SUPABASE_SERVICE_ROLE_KEY(또는 anon 키) 설정이 필요합니다.")
    st.stop()

maybe_bootstrap(supabase, st.session_state.user_id)

if not st.session_state.user_id:
    st.info(
        "로그인 전에: **`app_users` 없음(PGRST205)** 이면 `fix_app_users_only.sql` 만 SQL Editor에서 실행. "
        "전체 초기화는 **`run_all_schema_ordered.sql`** 한 파일을 순서대로 모두 실행(맨 위는 rag_* 데이터 삭제). "
        "또는 `multi-session-ref.sql` → `create_app_users_schema.sql` 각각 실행. "
        "`.env`에 **`SUPABASE_URL`**·**`SUPABASE_SERVICE_ROLE_KEY`** 후 사이드바 **회원가입** → 로그인."
    )

with st.sidebar:
    st.markdown('<h2 style="color: #1f77b4;">API 키</h2>', unsafe_allow_html=True)
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", key="sb_openai_key")
    anthropic_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...", key="sb_anthropic_key")
    gemini_key = st.text_input("Google (Gemini) API Key", type="password", placeholder="AIza...", key="sb_gemini_key")
    ensure_api_keys(openai_key, anthropic_key, gemini_key)

    _sb_key_hint = _supabase_key_warning()
    if _sb_key_hint:
        st.warning(_sb_key_hint)

    st.markdown('<h2 style="color: #1f77b4;">로그인</h2>', unsafe_allow_html=True)
    st.caption("Supabase Auth 미사용 · 앱 DB(`app_users`)에 저장된 ID/비밀번호")
    login_id = st.text_input("사용자 ID", key="sb_login_id")
    login_pw = st.text_input("비밀번호", type="password", key="sb_login_pw")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("로그인", use_container_width=True):
            if login_id and login_pw:
                if sign_in(supabase, login_id, login_pw):
                    st.success("로그인되었습니다.")
                    st.rerun()
            else:
                st.warning("사용자 ID와 비밀번호를 입력하세요.")
    with c2:
        if st.button("로그아웃", use_container_width=True):
            sign_out()
            st.success("로그아웃했습니다.")
            st.rerun()

    if st.button("회원가입 (별도 페이지)", use_container_width=True):
        st.switch_page("pages/_signup.py")

    if st.session_state.user_id:
        st.info(f"로그인: **{st.session_state.user_id}**")
    else:
        st.warning("로그인 후 세션·PDF·저장 기능을 사용할 수 있습니다.")

    st.markdown('<h2 style="color: #1f77b4;">LLM 선택</h2>', unsafe_allow_html=True)
    _models = ["gpt-5.5", "claude-opus-4-7", "gemini-3-pro-preview"]
    _mi = _models.index(st.session_state.selected_model) if st.session_state.selected_model in _models else 0
    st.session_state.selected_model = st.selectbox("모델 선택", _models, index=_mi)

    uid = st.session_state.user_id
    sessions = fetch_sessions(supabase, uid) if uid else []
    mapping = {f"{s['title']} ({s['updated_at'][:16]})": s["id"] for s in sessions}
    st.markdown('<h2 style="color: #1f77b4;">세션 리스트</h2>', unsafe_allow_html=True)
    selected_label = st.selectbox("세션 리스트 메뉴", ["(선택 안 함)"] + list(mapping.keys()))
    selected_session_id = mapping.get(selected_label)
    st.session_state.selected_session_id = selected_session_id

    disabled_sess = not uid
    if st.button("세션저장", use_container_width=True, disabled=disabled_sess):
        upsert_session(supabase, uid)
        clear_screen()
        st.session_state.current_session_id = str(uuid.uuid4())
        st.success("세션을 저장하고 화면을 초기화했습니다.")
        st.rerun()

    if st.button("세션로드", use_container_width=True, disabled=disabled_sess):
        if selected_session_id:
            if st.session_state.chat_history:
                upsert_session(supabase, uid)
            if load_session(supabase, selected_session_id, uid):
                st.success("선택한 세션을 불러왔습니다.")
                st.rerun()
        else:
            st.warning("세션 리스트에서 세션을 먼저 선택하세요.")

    if st.button("세션삭제", use_container_width=True, disabled=disabled_sess):
        target = selected_session_id or st.session_state.current_session_id
        if target:
            delete_session(supabase, target, uid)
            st.success("세션과 관련 데이터가 삭제되었습니다.")
            st.rerun()

    if st.button("화면초기화", use_container_width=True, disabled=disabled_sess):
        clear_screen()
        st.success("화면 데이터만 초기화했습니다.")
        st.rerun()

    if st.button("제목보정", use_container_width=True, disabled=disabled_sess):
        if st.session_state.current_session_id and st.session_state.chat_history:
            new_title = generate_session_title()
            supabase.table("rag_sessions").update({"title": new_title}).eq("id", st.session_state.current_session_id).eq(
                "user_id", uid
            ).execute()
            st.success(f"제목을 저장했습니다: {new_title}")
            st.rerun()
        else:
            st.warning("저장할 대화 또는 세션이 없습니다.")

    if st.button("vectordb", use_container_width=True, disabled=disabled_sess):
        sources = st.session_state.active_sources
        if sources:
            st.info("현재 vectordb 파일명:\n" + "\n".join(sorted(set(sources))))
        else:
            st.warning("현재 세션에 연결된 vectordb 파일이 없습니다.")

    st.markdown("---")
    uploaded_files = st.file_uploader("PDF 파일을 선택하세요", type=["pdf"], accept_multiple_files=True)
    if st.button("파일 처리하기", use_container_width=True, disabled=disabled_sess):
        if not uploaded_files:
            st.warning("먼저 PDF 파일을 업로드하세요.")
        elif not os.getenv("OPENAI_API_KEY"):
            st.warning("OpenAI API 키를 사이드바에서 입력하세요.")
        else:
            if not st.session_state.current_session_id:
                st.session_state.current_session_id = str(uuid.uuid4())
            ensure_session_row(
                supabase,
                st.session_state.current_session_id,
                st.session_state.selected_model,
                uid,
            )
            embeddings = get_embeddings()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            total_added = 0
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getbuffer())
                    tmp_path = tmp.name
                docs = PyPDFLoader(tmp_path).load()
                for d in docs:
                    d.metadata = d.metadata or {}
                    d.metadata["source"] = file.name
                chunks = splitter.split_documents(docs)
                added = save_chunks_to_vector_db(
                    supabase, embeddings, st.session_state.current_session_id, chunks, file.name, uid
                )
                total_added += added
                os.remove(tmp_path)
                if file.name not in st.session_state.active_sources:
                    st.session_state.active_sources.append(file.name)
                if file.name not in st.session_state.processed_files:
                    st.session_state.processed_files.append(file.name)
            st.session_state.retriever_ready = True
            st.success(f"파일 처리 완료: 신규 임베딩 {total_added}개 저장, 기존 임베딩은 재사용했습니다.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

logged_in = bool(st.session_state.user_id)
question = st.chat_input("질문을 입력하세요", disabled=not logged_in)

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    uid_chat = st.session_state.user_id
    if not st.session_state.retriever_ready or not st.session_state.current_session_id:
        warning = "먼저 PDF를 처리하거나 세션을 로드해 주세요."
        with st.chat_message("assistant"):
            st.markdown(warning)
        st.session_state.chat_history.append({"role": "assistant", "content": warning})
    else:
        embeddings = get_embeddings()
        docs = retrieve_docs(
            supabase, embeddings, st.session_state.current_session_id, uid_chat, question, k=5
        )
        context = "\n\n".join([d.page_content for d in docs[:3]])
        llm = build_llm(st.session_state.selected_model)
        prompt = (
            "당신은 PDF 기반 멀티유저 RAG 챗봇입니다.\n"
            f"질문: {question}\n"
            f"문서 컨텍스트:\n{context}\n"
            "문서 기반으로 정확히 답하고, 추측은 피하세요."
        )

        answer = ""
        with st.chat_message("assistant"):
            ph = st.empty()
            for chunk in llm.stream(prompt):
                chunk_text = getattr(chunk, "content", str(chunk))
                answer += chunk_text
                ph.markdown(answer + "▌")

            followups = generate_followups(question, answer)
            if followups:
                answer += "\n\n### 💡 다음에 필요한 질문 3개\n"
                for idx, item in enumerate(followups, 1):
                    answer += f"\n{idx}. {item}"
            ph.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
