import hashlib
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

load_dotenv()

st.set_page_config(page_title="PDF 기반 멀티세션 RAG 챗봇", page_icon="📚", layout="wide")

st.markdown(
    """
<style>
h1 { font-size: 1.4rem !important; font-weight: 600 !important; color: #ff69b4 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #ffd700 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1f77b4 !important; }
.stChatMessage { font-size: 0.95rem !important; line-height: 1.5 !important; }
.stChatMessage * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; }
.stButton > button { background-color: #ff69b4 !important; color: white !important; border: none !important; border-radius: 5px !important; font-weight: bold !important; }
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
    <span style="color: #9b59b6;">기반 멀티세션</span>
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


@st.cache_resource
def get_supabase() -> Optional[Client]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def get_embeddings() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 필요합니다.")
    return OpenAIEmbeddings(openai_api_key=api_key)


def build_llm(model_name: str) -> Any:
    if model_name == "gpt-5.5":
        return ChatOpenAI(model="gpt-5.5", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
    if model_name == "claude-opus-4-7":
        return ChatAnthropic(model="claude-opus-4-7", temperature=0.7, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
    if model_name == "gemini-3-pro-preview":
        return ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.7, google_api_key=os.getenv("GOOGLE_API_KEY"))
    raise ValueError(f"지원하지 않는 모델: {model_name}")


def fetch_sessions(supabase: Client) -> List[Dict[str, Any]]:
    res = supabase.table("rag_sessions").select("*").order("updated_at", desc=True).limit(200).execute()
    return res.data or []


def generate_session_title() -> str:
    user_msg = next((m["content"] for m in st.session_state.chat_history if m["role"] == "user"), "").strip()
    ai_msg = next((m["content"] for m in st.session_state.chat_history if m["role"] == "assistant"), "").strip()
    if not user_msg:
        return "새 세션"
    try:
        llm = ChatOpenAI(model="gpt-5.5", temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"질문과 답변 핵심으로 한국어 세션 제목 18자 이내 1개만 출력.\n질문:{user_msg[:500]}\n답변:{ai_msg[:700]}"
        title = str(llm.invoke(prompt).content).strip().replace("\n", " ")
        return title[:30] if title else user_msg[:30]
    except Exception:
        return user_msg[:30]


def upsert_session(supabase: Client) -> bool:
    if not st.session_state.current_session_id:
        st.session_state.current_session_id = str(uuid.uuid4())
    title = generate_session_title()
    payload = {
        "id": st.session_state.current_session_id,
        "title": title,
        "model_name": st.session_state.selected_model,
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


def ensure_session_row(supabase: Client, session_id: str, model_name: str) -> None:
    """rag_sessions FK를 만족하도록 세션 행을 선생성/보장."""
    if not session_id:
        raise ValueError("session_id가 비어 있습니다.")
    supabase.table("rag_sessions").upsert(
        {
            "id": session_id,
            "title": "새 세션",
            "model_name": model_name,
        }
    ).execute()


def load_session(supabase: Client, session_id: str) -> bool:
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
    return True


def clear_screen() -> None:
    st.session_state.chat_history = []
    st.session_state.processed_files = []
    st.session_state.active_sources = []
    st.session_state.retriever_ready = False


def delete_session(supabase: Client, session_id: str) -> bool:
    supabase.table("rag_sessions").delete().eq("id", session_id).execute()
    if st.session_state.current_session_id == session_id:
        st.session_state.current_session_id = None
        clear_screen()
    return True


def save_chunks_to_vector_db(
    supabase: Client, embeddings: OpenAIEmbeddings, session_id: str, chunks: List[Document], source_name: str
) -> int:
    unique_chunks: List[Document] = []
    for ch in chunks:
        text = str(ch.page_content).strip()
        if not text:
            continue
        hash_value = hashlib.sha256(text.encode("utf-8")).hexdigest()
        exists = supabase.table("rag_documents").select("id").eq("content_hash", hash_value).limit(1).execute()
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


def retrieve_docs(supabase: Client, embeddings: OpenAIEmbeddings, session_id: str, query: str, k: int = 5) -> List[Document]:
    query_vec = embeddings.embed_query(query)
    res = supabase.rpc(
        "match_rag_documents",
        {
            "query_embedding": query_vec,
            "target_session_id": session_id,
            "match_count": k,
            "match_threshold": 0.6,
        },
    ).execute()
    docs: List[Document] = []
    for row in res.data or []:
        docs.append(Document(page_content=row["content"], metadata=row.get("metadata", {}) or {}))
    return docs


def generate_followups(question: str, answer: str) -> List[str]:
    try:
        llm = ChatOpenAI(model="gpt-5.5", temperature=0.8, openai_api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"질문:{question}\n답변:{answer[:1200]}\n다음에 물어볼 질문 3개만 줄바꿈으로 출력."
        text = str(llm.invoke(prompt).content).strip()
        items = [x.strip("-• 1234567890. ").strip() for x in text.split("\n") if x.strip()]
        return items[:3]
    except Exception:
        return []


supabase = get_supabase()
if not supabase:
    st.error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY(SUPABASE_ANON_KEY) 설정이 필요합니다.")
    st.stop()

if not st.session_state.bootstrapped:
    sessions = fetch_sessions(supabase)
    if sessions:
        load_session(supabase, sessions[0]["id"])
        st.session_state.selected_session_id = sessions[0]["id"]
    else:
        st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.bootstrapped = True

with st.sidebar:
    st.markdown('<h2 style="color: #1f77b4;">LLM 선택</h2>', unsafe_allow_html=True)
    st.session_state.selected_model = st.selectbox(
        "모델 선택",
        ["gpt-5.5", "claude-opus-4-7", "gemini-3-pro-preview"],
        index=["gpt-5.5", "claude-opus-4-7", "gemini-3-pro-preview"].index(st.session_state.selected_model),
    )

    sessions = fetch_sessions(supabase)
    mapping = {f"{s['title']} ({s['updated_at'][:16]})": s["id"] for s in sessions}
    st.markdown('<h2 style="color: #1f77b4;">세션 리스트</h2>', unsafe_allow_html=True)
    selected_label = st.selectbox("세션 리스트 메뉴", ["(선택 안 함)"] + list(mapping.keys()))
    selected_session_id = mapping.get(selected_label)
    st.session_state.selected_session_id = selected_session_id

    if st.button("세션저장", use_container_width=True):
        upsert_session(supabase)
        clear_screen()
        st.session_state.current_session_id = str(uuid.uuid4())
        st.success("세션을 저장하고 화면을 초기화했습니다.")
        st.rerun()

    if st.button("세션로드", use_container_width=True):
        if selected_session_id:
            if st.session_state.chat_history:
                upsert_session(supabase)
            load_session(supabase, selected_session_id)
            st.success("선택한 세션을 불러왔습니다.")
            st.rerun()
        else:
            st.warning("세션 리스트에서 세션을 먼저 선택하세요.")

    if st.button("세션삭제", use_container_width=True):
        target = selected_session_id or st.session_state.current_session_id
        if target:
            delete_session(supabase, target)
            st.success("세션과 관련 데이터가 삭제되었습니다.")
            st.rerun()

    if st.button("화면초기화", use_container_width=True):
        clear_screen()
        st.success("화면 데이터만 초기화했습니다.")
        st.rerun()

    if st.button("vectordb", use_container_width=True):
        sources = st.session_state.active_sources
        if sources:
            st.info("현재 vectordb 파일명:\n" + "\n".join(sorted(set(sources))))
        else:
            st.warning("현재 세션에 연결된 vectordb 파일이 없습니다.")

    st.markdown("---")
    uploaded_files = st.file_uploader("PDF 파일을 선택하세요", type=["pdf"], accept_multiple_files=True)
    if st.button("파일 처리하기", use_container_width=True):
        if not uploaded_files:
            st.warning("먼저 PDF 파일을 업로드하세요.")
        else:
            if not st.session_state.current_session_id:
                st.session_state.current_session_id = str(uuid.uuid4())
            ensure_session_row(
                supabase,
                st.session_state.current_session_id,
                st.session_state.selected_model,
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
                added = save_chunks_to_vector_db(supabase, embeddings, st.session_state.current_session_id, chunks, file.name)
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

question = st.chat_input("질문을 입력하세요")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not st.session_state.retriever_ready or not st.session_state.current_session_id:
        warning = "먼저 PDF를 처리하거나 세션을 로드해 주세요."
        with st.chat_message("assistant"):
            st.markdown(warning)
        st.session_state.chat_history.append({"role": "assistant", "content": warning})
    else:
        embeddings = get_embeddings()
        docs = retrieve_docs(supabase, embeddings, st.session_state.current_session_id, question, k=5)
        context = "\n\n".join([d.page_content for d in docs[:3]])
        llm = build_llm(st.session_state.selected_model)
        prompt = (
            "당신은 PDF 기반 멀티세션 RAG 챗봇입니다.\n"
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

