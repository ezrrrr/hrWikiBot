import os, re, base64, textwrap
from typing import List, Dict, Any
from dotenv import load_dotenv
import streamlit as st

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

load_dotenv()

SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_KEY = (
    os.getenv("AOAI_KEY")
    or os.getenv("AZURE_OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
AOAI_VERSION = (
    os.getenv("AOAI_VERSION") or os.getenv("OPENAI_API_VERSION") or "2024-02-15-preview"
)
AOAI_DEPLOYMENT = os.getenv("AOAI_DEPLOYMENT") or "gpt-4o-mini"

REQUIRED = {
    "SEARCH_ENDPOINT": SEARCH_ENDPOINT,
    "SEARCH_API_KEY": SEARCH_API_KEY,
    "INDEX_NAME": INDEX_NAME,
    "AOAI_ENDPOINT": AOAI_ENDPOINT,
    "AOAI_KEY": AOAI_KEY,
    "AOAI_VERSION": AOAI_VERSION,
    "AOAI_DEPLOYMENT": AOAI_DEPLOYMENT,
}
missing = [k for k, v in REQUIRED.items() if not v]

top_k = 2
use_highlight = False
use_semantic = False
sem_config = None
max_ctx = 1000
temperature = 0.7


APP_TITLE = "HRWikiBot - 사내 가이드북 챗봇"
CATEGORIES = [
    "HR/인사",
    "근로시간·휴가",
    "급여·복리후생",
    "경조사 지원",
    "교육·온보딩",
    "업무 프로세스",
    "안전·윤리·준법",
    "FAQ",
]

# ---------- 초기 설정 ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🗂️", layout="wide")

# ---------- 사이드바 ----------
with st.sidebar:
    st.title("카테고리")

    for cat in CATEGORIES:
        st.subheader("- " + cat)
    st.markdown("---")
    top_k = st.slider("검색 문서 수 (Top-K)", 1, 8, 3)

    # ---------- 푸터 ----------
    st.markdown(
        """
        <style>
        .footer {position: fixed;left: 0;bottom: 0;width: 100%;background-color: #f5f5f5;color: gray;text-align: center;padding: 8px;font-size: 13px;border-top: 1px solid #ddd; z-index:9999}
        </style>
        <div class="footer">
            © 2025 HRWikiBot | 내부용 상담 서비스
        </div>
    """,
        unsafe_allow_html=True,
    )


# ---------- 클라이언트 (캐시) ----------
@st.cache_resource(show_spinner=False)
def get_clients():
    if missing:
        return None, None
    sc = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )
    ao = AzureOpenAI(
        api_key=AOAI_KEY,
        azure_endpoint=AOAI_ENDPOINT,
        api_version=AOAI_VERSION,
    )
    return sc, ao


search_client, aoai_client = get_clients()

# ---------- 유틸 ----------
SEARCH_FIELDS = ["merged_content", "content", "text", "layoutText", "translated_text"]
SELECT_FIELDS = [
    "metadata_storage_name",
    "metadata_storage_path",
    "metadata_storage_last_modified",
    "merged_content",
    "content",
    "text",
    "layoutText",
    "translated_text",
]


def clean_text(t: str, max_len=1000) -> str:
    if not isinstance(t, str):
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    return (t[:max_len] + "…") if len(t) > max_len else t


def pick_body(d: Dict[str, Any]) -> str:
    for k in SEARCH_FIELDS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def decode_blob_path(p: str) -> str:
    try:
        return base64.b64decode(p).decode("utf-8")
    except Exception:
        return p or ""


def do_search(query: str, k: int) -> List[Dict[str, Any]]:
    if not search_client:
        return []
    kwargs = dict(
        search_text=query,
        select=SELECT_FIELDS,
        top=k,
        search_fields=SEARCH_FIELDS,  # 리스트!
        include_total_count=True,
    )
    if use_highlight:
        kwargs.update(
            {
                "highlight_fields": ",".join(SEARCH_FIELDS),  # 문자열(콤마)!
                "highlight_pre_tag": "<mark>",
                "highlight_post_tag": "</mark>",
            }
        )
    if use_semantic:
        kwargs.update({"query_type": "semantic", "semantic_configuration": sem_config})

    return list(search_client.search(**kwargs))


def build_context(docs: List[Dict[str, Any]], limit_chars: int) -> str:
    total = 0
    blocks = []
    for i, d in enumerate(docs, 1):
        hl = d.get("@search.highlights") or {}
        if hl:
            snips = []
            for _, arr in hl.items():
                snips.extend(arr[:2])
            snippet = clean_text(" … ".join(snips), 1600)
        else:
            snippet = clean_text(pick_body(d), 1600)
        if not snippet:
            continue
        if total + len(snippet) > limit_chars:
            break
        total += len(snippet)
        name = d.get("metadata_storage_name") or f"doc-{i}"
        blocks.append(f"[{i}] {name}\n{snippet}")
    return "\n\n---\n\n".join(blocks)


def ask_rag(query: str, docs: List[Dict[str, Any]]) -> str:
    ctx = build_context(docs, max_ctx)
    if not ctx:
        return "관련 문서 컨텍스트가 없습니다. 질문을 다르게 해보거나 인덱스를 확인해주세요."
    sys_prompt = textwrap.dedent(
        """\
너는 HR 규정 도우미 챗봇이야.
다음 문서를 참고하여 사용자의 질문에 답해줘.
    """
    ).strip()
    user_prompt = f"# 문서 발췌\n{ctx}\n\n# 질문\n{query}"
    resp = aoai_client.chat.completions.create(
        model=AOAI_DEPLOYMENT,  # 배포 이름
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


# ---- 컨텐츠 2단 레이아웃
main, right = st.columns([0.66, 0.36], gap="large")
# ---------- UI ----------
# Main
with main:
    st.title("🗂️ HRWikiBot – 신입사원을 위한 HR,규정, 복지 챗봇 ")

    q = st.text_input(
        "질문을 입력하세요",
        "아기를 임신했어요 임신과 출산 관련 복지가 있나요?",
    )

    btn_search = st.button("검색")


if (btn_search) and missing:
    st.stop()

if btn_search:
    try:
        with st.spinner("검색 중…"):
            docs = do_search(q, top_k)

        # LAG 모델 답변 생성
        with st.spinner("모델이 답변을 생성 중…"):
            answer = ask_rag(q, docs)
        st.subheader("💬 답변")
        st.markdown(answer)

        # 참고 문서 간단 출력
        refs = []

        if docs:
            d = docs[0]  # 첫 번째 문서만 가져옴
            refs.append(f"[1] {d.get('metadata_storage_name')}")
        if refs:
            st.markdown("**참고 문서:**\n" + "\n".join(refs))

        st.subheader(f"검색 결과 ({len(docs)}건)")
        if not docs:
            st.info("검색 결과가 없습니다. 질문을 바꿔보거나 인덱스를 확인해 주세요.")
        for i, d in enumerate(docs, 1):
            fname = d.get("metadata_storage_name")
            lm = d.get("metadata_storage_last_modified")
            path = decode_blob_path(d.get("metadata_storage_path"))
            with st.expander(f"[{i}] {fname}  |  {lm}", expanded=(i == 1)):
                if use_highlight and d.get("@search.highlights"):
                    st.markdown("**하이라이트**")
                    for f, snips in d["@search.highlights"].items():
                        st.markdown(
                            f"- *{f}*: " + " … ".join(snips[:2]), unsafe_allow_html=True
                        )
                else:
                    st.markdown(clean_text(pick_body(d), 500))

    except Exception as e:
        st.error(f"오류: {e}")
        st.exception(e)


# Right panel
with right:
    st.subheader("보조 정보")
    tabs = st.tabs(["📝슬기로운 신입생활", "📊 인기 복지 TOP5"])
    with tabs[0]:
        st.markdown(" 💬 첫출근을 했어요")
        st.markdown(" 💬 복지몰과 복지포인트")
        st.markdown(" 💬 점심 식사 관련")
        st.markdown(" 💬 비혼선언")
    with tabs[1]:
        st.markdown(" 🧾 가족 결혼 복지")
        st.markdown(" 🧾 사내 봉사활동")
        st.markdown(" 🧾 가족이 돌아가셨어")
        st.markdown(" 🧾 사내 동호회")
