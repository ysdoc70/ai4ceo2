import hashlib
import os
import secrets
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from postgrest.exceptions import APIError
from supabase import create_client

PBKDF2_ITERS = 390_000


def hash_password_new(password: str) -> tuple[str, str]:
    salt_bytes = secrets.token_bytes(16)
    salt_hex = salt_bytes.hex()
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERS)
    return dk.hex(), salt_hex


_APP_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _APP_DIR.parent.parent
load_dotenv(_APP_DIR / ".env")
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()


def _sync_streamlit_secrets_to_env() -> None:
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


_sync_streamlit_secrets_to_env()

st.set_page_config(page_title="회원가입", page_icon="✉️", layout="wide")

st.markdown(
    """
<style>
h1 { font-size: 1.5rem !important; font-weight: 600 !important; color: #ff69b4 !important; }
.stButton > button { background-color: #ff69b4 !important; color: white !important; border: none !important;
  border-radius: 5px !important; font-weight: bold !important; }
.stButton > button:hover { background-color: #ff1493 !important; }
</style>
""",
    unsafe_allow_html=True,
)


def _db_key() -> str:
    return (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
        or os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
    )


url = os.getenv("SUPABASE_URL")
key = _db_key()

if not url or not key:
    st.error("SUPABASE_URL 및 SUPABASE_SERVICE_ROLE_KEY(또는 anon 키)를 Secrets 또는 .env에 설정하세요.")
    st.stop()

if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
    st.warning("`app_users` 가입은 **`SUPABASE_SERVICE_ROLE_KEY`** 가 있을 때 가장 안정적으로 동작합니다.")

client = create_client(url, key)

st.markdown("## 회원가입")
st.caption("사용자 ID와 비밀번호만 등록합니다. (Supabase Auth 미사용)")

username = st.text_input("사용자 ID", key="signup_user")
pw = st.text_input("비밀번호", type="password", key="signup_pw")
pw2 = st.text_input("비밀번호 확인", type="password", key="signup_pw2")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("가입 요청", use_container_width=True):
        u = username.strip()
        if not u or len(u) > 128:
            st.warning("사용자 ID는 1~128자여야 합니다.")
        elif not pw:
            st.warning("비밀번호를 입력하세요.")
        elif pw != pw2:
            st.error("비밀번호가 일치하지 않습니다.")
        else:
            try:
                ph, sh = hash_password_new(pw)
                client.table("app_users").insert({"username": u, "password_hash": ph, "salt": sh}).execute()
                st.success("가입되었습니다. 메인 화면에서 같은 ID로 로그인하세요.")
            except APIError as e:
                msg = getattr(e, "message", str(e))
                if "duplicate" in msg.lower() or "23505" in msg:
                    st.error("이미 사용 중인 사용자 ID입니다.")
                else:
                    st.error(f"가입 오류: {msg}")
            except Exception as e:
                st.error(f"가입 오류: {e}")

with col_b:
    if st.button("메인으로", use_container_width=True):
        st.switch_page("multi-users-ref.py")
