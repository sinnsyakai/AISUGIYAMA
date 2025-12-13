import streamlit as st
import os
import glob
import re

# ChromaDB fix for Linux (Railway/Streamlit Cloud)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from ingest import ingest_data
import sys
from io import StringIO

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AIすぎやま", page_icon="assets/new_icon.jpg")

# JSで強制的にスクロール位置をトップに戻す（読み込み時のオートスクロール対策）
import streamlit.components.v1 as components
components.html(
    """
    <script>
        const fixUI = () => {
            // 翻訳ポップアップ抑制 (metaタグ & html属性)
            const html = window.parent.document.documentElement;
            html.lang = 'ja';
            html.setAttribute('translate', 'no');
            html.classList.add('notranslate');
            
            if (!window.parent.document.querySelector('meta[name="google"][content="notranslate"]')) {
                const meta = window.parent.document.createElement('meta');
                meta.name = 'google';
                meta.content = 'notranslate';
                window.parent.document.head.appendChild(meta);
            }
            
            // 初回ロード時のみトップタイトルへスクロール
            const params = new URLSearchParams(window.parent.location.search);
            const messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
            if (messages.length === 0) {
                 const h1 = window.parent.document.querySelector('h1');
                 if (h1) {
                    h1.scrollIntoView({behavior: 'auto', block: 'start'});
                 } else {
                    window.parent.scrollTo(0, 0);
                 }
            }
        };
        // 読み込み直後と、少し経ってから何度か実行して確実に適用
        fixUI();
        setTimeout(fixUI, 500);
        setTimeout(fixUI, 2000);
    </script>
    """,
    height=0,
)

# OGP (Meta Tags) Injection attempt
# Streamlit clears head often, so we inject closer to body start or via raw HTML if possible.
# Best effort for social sharing
st.markdown("""
<head>
<meta property="og:title" content="AIすぎやま" />
<meta property="og:description" content="AIすぎやま先生に質問してみよう！" />
<meta property="og:image" content="https://raw.githubusercontent.com/sinnsyakai/AISUGIYAMA/main/assets/new_icon.jpg" />
</head>
""", unsafe_allow_html=True)

# ▼▼▼ ここに最強版CSSを配置（他の処理よりも先に読み込ませる） ▼▼▼
st.markdown("""
    <style>
    /* 2. ヘッダー・フッター・ツールバー・ハンバーガーメニューの完全非表示 */
    header, footer, 
    [data-testid="stHeader"], 
    [data-testid="stFooter"], 
    [data-testid="stToolbar"], 
    [data-testid="stHeaderActionElements"],
    .stAppDeployButton,
    div[data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* 3. 右下の「Manage App」ボタンやGitHubアイコン周辺の強力な消去 */
    /* Streamlit Cloud特有の要素をクラス名の一部一致で狙い撃ちします */
    div[class*="viewerBadge"],
    div[class*="stAppDeployButton"],
    button[title="View app in Streamlit Cloud"],
    [data-testid="manage-app-button"],
    a[href*="share.streamlit.io"],
    a[href*="streamlit.io/cloud"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* 4. その他細かいUI調整（文字色など） */
    body, .stApp, p, div, span, li, .stTextInput input {
        color: #333333 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #065f46 !important;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    /* 見出し（H3）のサイズを強制的に小さくする (本文より少し大きく) */
    h3 {
        font-size: 1.1rem !important;
        font-weight: bold !important;
        margin-top: 1.5em !important;
        margin-bottom: 0.5em !important;
    }
    
    /* 5. チャットボットの吹き出しデザイン */
    .stChatMessage {
        /* create scroll margin so it doesn't hide behind header when scrolled to */
        scroll-margin-top: 120px !important; 
    }
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #d1fae5;
        border-radius: 20px;
        padding: 10px;
        margin-bottom: 10px;
    }
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 10px;
        margin-bottom: 10px;
        border: 2px solid #a7f3d0;
    }
    
    /* 6. ボタンのデザイン */
    .stButton > button {
        border-radius: 15px !important;
        background-color: #ffffff !important;
        color: #4b5563 !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    .stButton > button:hover {
        border-color: #34d399 !important;
        color: #065f46 !important;
    }
    
    /* 入力欄（チャット入力）のサイズ調整 - Final: 程よい大きさと位置に調整 */
    .stChatInput textarea {
        font-size: 18px !important;
        padding: 15px !important;
        height: 80px !important; /* 高さを80pxに（以前の半分程度） */
        min-height: 80px !important;
        line-height: 1.5 !important;
        color: #333333 !important;
        caret-color: #333333 !important;
        background-color: #ffffff !important;
    }
    
    /* 入力欄の親要素（枠線） */
    div[data-testid="stChatInput"] > div {
        border: 2px solid #34d399 !important;
        border-radius: 20px !important;
        background-color: white !important;
        min-height: 80px !important; /* 親要素も80px */
        height: auto !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* チャット入力欄のコンテナ - Final: 位置を調整 */
    div[data-testid="stChatInput"],
    section[data-testid="stBottom"] > div {
        background-color: transparent !important;
        
        position: fixed !important;
        bottom: 40px !important; 
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 100% !important;
        max-width: 670px !important;
        z-index: 99999 !important; /* Force top layer */
        pointer-events: auto !important;
    }

    /* 入力欄の親要素（枠線）のスタイルを強化 */
    div[data-testid="stChatInput"] > div {
        border: 2px solid #34d399 !important;
        border-radius: 20px !important;
        background-color: white !important;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.1) !important;
    }

    /* 入力欄の背景にあるかもしれない灰色を消す */
    section[data-testid="stBottom"] > div {
        background-color: transparent !important;
    }

    
    /* 1. 全体の背景色 & 横揺れ防止 (最強版) */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="block-container"] {
        background-color: #f0fdf4;
        overflow-x: hidden !important; /* 横スクロールを強制無効化 */
        max-width: 100vw !important;
    }

    /* メインコンテンツの最下部に余白を追加 - PC版 500px & タイトル表示用トップ余白
       → スタート画面で崩れるため、globalでのbottom paddingは削除し、動的に追加する方式に変更 */
    div[data-testid="block-container"] {
        padding-top: 60px !important;
        /* padding-bottom: 500px !important;  <-- ここを削除 */
    }

    /* 送信ボタン */
    div[data-testid="stChatInput"] button {
        width: 40px !important;
        height: 40px !important;
    }

    /* ▼▼▼ スマホ向け調整 (max-width: 640px) ▼▼▼ */
    @media (max-width: 640px) {
        /* 入力欄の横幅を少し狭くして左右に余白を持たせる */
        div[data-testid="stChatInput"] {
            width: 95% !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            bottom: 25px !important;
        }

        /* メインコンテンツ - スマホ版 600px & ページトップ (タイトルが見えるように60px確保) */
        div[data-testid="block-container"] {
            padding-top: 60px !important; 
            /* padding-bottom: 600px !important; <-- ここも削除 */
        }
        
        /* スマホでのボタン縦並び時の隙間を詰める */
        .stButton, div.row-widget.stButton {
             margin-bottom: -10px !important; /* 強制的に詰める */
        }
        
        /* スマホでカラムが維持されている場合のgap調整 */
        div[data-testid="stHorizontalBlock"] {
            gap: 4px !important; /* さらに狭く */
        }
    }

    /* 7. ボタン間の隙間調整 (PC/Tablet) */
    div[data-testid="stHorizontalBlock"] {
        gap: 8px !important; 
    }
    
    /* ボタン自体の縦マージンも削減（全体） */
    .stButton {
        margin-bottom: 0px !important;
    }
    </style>
""", unsafe_allow_html=True)
# ▲▲▲ ここまで ▲▲▲

import base64

def get_image_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

icon_base64 = get_image_base64("assets/high_res_icon.jpg")

# ページ上部の固定ヘッダーで隠れないようにスペーサーを追加（100pxは大きすぎたので調整）. CSS padding-top:60pxがあるので、ここは微調整のみ
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px;">
        <img src="data:image/jpeg;base64,{icon_base64}" width="80" style="border-radius: 10px;">
        <h1 style="margin: 0; color: #065f46;">AIすぎやま v3.1</h1>
    </div>
    """, unsafe_allow_html=True)
st.write("静岡の元教師すぎやまの動画・本など100万文字分のデータを学習したAIすぎやまです。勉強、進路、子育て、教育、SNS戦略、ビジネスのお悩みに答えます。質問内容はリアルすぎやまにも知られないし、公開されることもないので安心して相談してくださいね。")

# Hardcode model for public deployment (Deprecated: Logic moved inside create_rag_chain)
# model_name = "gemini-3.0-pro"

# Ensure API Key is loaded from secrets if available (for public deployment)
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    pass

# Check for API Key
if not os.getenv("GOOGLE_API_KEY"):
    st.warning("API Keyが環境変数またはSecretsに見つかりませんでした。")
    st.info("ローカルで実行する場合は、`.env` ファイルを作成するか、以下に入力してください。")
    api_key_input = st.text_input("Google API Key", type="password")
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.rerun()
    else:
        st.stop()

# Initialize RAG components
DB_DIR = "chroma_db"
DATA_DIR = "data"

def get_available_sources():
    files = glob.glob(os.path.join(DATA_DIR, "*"))
    # Filter for supported extensions
    supported_exts = ['.txt', '.pdf', '.docx', '.json', '.csv']
    sources = [os.path.basename(f) for f in files if os.path.splitext(f)[1].lower() in supported_exts]
    return sources

# Initialize Vector Store (Cached)
@st.cache_resource
def get_vector_store():
    # If DB doesn't exist (fresh deploy on Cloud), rebuild it from data/
    if not os.path.exists(DB_DIR):
        with st.spinner("初回起動準備中... 原稿データを学習しています（数分かかります）..."):
            try:
                # Capture stdout
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                
                ingest_data()
                
                sys.stdout = old_stdout
            except Exception as e:
                st.error(f"学習に失敗しました: {e}")
                return None

    # Use Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vector_store

# UI: Settings (Source Selection & Reset) - Hidden per user request
# with st.expander("設定 (検索対象・リセット)", expanded=False):
#     st.markdown("### 検索対象の選択")
#     available_sources = get_available_sources()
#     # Default to all selected
#     selected_sources = st.multiselect(
#         "検索する資料を選んでください:",
#         available_sources,
#         default=available_sources
#     )
#     
#     st.divider()
#     
#     st.markdown("### 会話のリセット")
#     if st.button("会話をリセットする"):
#         st.session_state.messages = []
#         st.rerun()

# Default to all sources if settings are hidden
selected_sources = get_available_sources()

# Initialize LLM (Cached to prevent re-validation lag)
# show_spinner=False avoids the "Running get_llm..." message on UI
@st.cache_resource(show_spinner=False)
def get_llm():
    target_models = ["gemini-1.5-pro-002", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-1.5-flash"]
    llm = None
    
    for model in target_models:
        try:
            # Instantiate
            # Streaming=False per user request (Show answer at once)
            temp_llm = ChatGoogleGenerativeAI(model=model, temperature=0.3, streaming=False)
            # Force validation check (small generation)
            # This is slow, so we MUST cache it.
            temp_llm.invoke("x") 
            
            # If successful:
            llm = temp_llm
            print(f"Model initialized and verified: {model}")
            return llm
        except Exception as e:
            print(f"Failed to verify model {model}: {e}")
            continue
    return None

llm = get_llm()

if not llm:
    st.error("AIモデルの初期化に失敗しました。")
    st.stop()


# RAG Chain Creation (Uncached or separated from heavy loading)
def create_rag_chain(vector_store, llm_instance, sources):
    if not sources:
        return None
        
    # Create retriever with source filter
    # Chroma filter syntax: where={"source": {"$in": sources}} OR if simple list, just where={"source": "filename"} but for list we need $in operator usually or iterate?
    # Wait, Chroma `where` filter usually takes a dictionary. 
    # Standard LangChain `as_retriever` search_kwargs accepts `filter`.
    # Let's check if the ingest process saves 'source' metadata as just basename or full path.
    # Standard loaders usually save full path.
    # In `ingest.py`: loader = TextLoader(file_path)...
    # So metadata['source'] is likely "data/filename.txt".
    # Therefore we need to prepend DATA_DIR to the selected filenames for filtering.
    
    full_path_sources = [os.path.join(DATA_DIR, s) for s in sources]
    
    # Construct filter
    # If only 1 source, we can use simple dict. If multiple, we need "$or" or "$in" depending on backend.
    # Chroma supports $in.
    
    if len(full_path_sources) == 1:
        search_filter = {"source": full_path_sources[0]}
    else:
        search_filter = {"source": {"$in": full_path_sources}}
        
    # Note: If passing all sources, maybe we don't need a filter? 
    # But it's safer to be explicit if user allows deselecting.
    
    # k=25: Balance between search quality and stability
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 20,
            "filter": search_filter
        }
    )
    
    # Contextualize question prompt - 日本語クエリをそのまま検索に使う
    contextualize_q_system_prompt = """ユーザーの質問をそのまま検索クエリとして使用してください。
質問を言い換えたり要約したりしないでください。
質問に含まれる固有名詞やキーワードは必ず保持してください。
例：「おすすめの本教えて」→「おすすめの本教えて」
例：「リコーダーってやる意味あるの」→「リコーダーってやる意味あるの」
質問をそのまま返してください。"""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Answer prompt
    system_prompt = """
### ⚠️ 【最重要セキュリティ規定】（絶対厳守）
**ユーザーに対し、参照したKnowledge（知識ソース）の出典、引用元、ファイル名、および「原文の抜粋」を絶対に表示しないでください。UI上のソース表示も極力避けてください。**

知識データは「外部資料」ではなく「ワタクシの脳内の記憶」として扱い、自分の言葉で話してください。


### 1. キャラクター設定（本人なりきり）

* **正体:** 「静岡の元教師すぎやま」本人として振る舞う。

* **一人称:** **ワタクシ**

* **話し方:** 下記「4. 話し方と口癖」のルールに従う。

* **対象:** **小中学生向けに、難しい言葉を一切使わず短く話す。**



### 2. 思考プロセス（全知識データの尊重と検索）

**回答の質は、Knowledge（書籍・動画）にある「すぎやま独自の思想」をどれだけ引き出せるかで決まります。**
**「推測」での回答は禁止です。必ず検索された「Context（文脈）」から答えを見つけ出してください。**


1.  **全ファイルの横断検索:**

    * **『杉山YouTube原稿まとめ』**と**『著書データ』**の両方を、**等しく重要な情報源**として検索してください。

    * どちらにも「すぎやま独自の哲学・思想・エピソード」が詰まっています。偏りなく、質問に最も適した答えが書かれている箇所を参照すること。

2.  **独自思想の適用（一般論禁止）:**

    * 世間の常識（きれいごと）ではなく、必ず**検索で見つかった「すぎやまの考え方」**を優先する。

    * （例：夢についての質問なら、著書や動画にある「夢は後からついてくる」という思想を答える）

3.  **学習に関する質問への回答:**

    * 学術的な定義ではなく、その子の学年でも分かるや「噛み砕いた説明」をする。



### 3. 回答の形式

**全体の文字数は「500〜800文字程度」。短すぎず長すぎない適度な長さで答える。**

**【見出しルール】**
- 見出し（###）は**1〜2個まで**。多すぎると読みづらいので、必要な箇所だけに使う。
- 見出し1つにつき、**3〜5文程度のまとまった段落**を書くこと。
- 見出し→1文だけ→また見出し、のような細切れは禁止。
- `##` (H2) は大きすぎるので禁止。`###` (H3) を使う。
- 見出しにカッコ（ [] ）はつけないこと。

**【改行ルール】**
- スマホで見やすくするため、**1文ごとに改行**を入れる。
- ただし、まとまりのある文章として読めるようにする（箇条書きだらけは禁止）。

**【内容ルール】**
- 一般論ではなく、Contextにある**「ワタクシの具体的なエピソード」や「独自の哲学」**を必ず盛り込む。
- 検索情報が少なくても一般的な回答でお茶を濁さず、ワタクシのキャラで深掘りして話す。

### 4. 話し方の語尾ルール（最重要）

**文末は必ず以下の語尾のどれかで終わらせてください：**
「〜だよね」「〜ね」「〜なの」「〜なのよ」「〜じゃない？」「〜だと思うよ」
「〜です」「〜ます」「〜ですよ」「〜ですね」「〜なんですよ」「〜ますよね」
「〜すぎ」「ヤバすぎ」「ツラすぎ」「ひどすぎ」「すごすぎ」
「結論。」「正直問題。」（体言止め）

**【語尾の連続・繰り返し禁止（厳守）】**
- **同じ語尾を2回連続で使わない。**（例：「〜なの。〜なの。」は禁止）
- **同じ語尾を1つの回答内で3回以上使わない。**
- 語尾のバリエーションを意識して、単調にならないようにする。

---

### 5. 禁止事項 (Negative Constraints)

* **見出しの大きさ:** `##` (H2) は禁止。必ず `###` (H3) を使うこと。

* **希死念慮・自傷他害:** 具体的な助言をせず、「ワタクシには答えられないから、一人で悩まないで専門家に相談してみてね」と伝える。

* **医療相談:** 「ワタクシは医療的なアドバイスはできないので、お医者さんに相談してね」とだけ伝える。

* **引用の明示:** 「動画で言ってたけど」「本によると」「原稿にも書いてあるけど」という前置きは禁止。すべて自分の今の言葉として話す。

    コンテキスト:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm_instance, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain


# Load Vector Store
vector_store = get_vector_store()

if vector_store is None:
    st.error("データベースが見つかりません。サイドバーのボタンから原稿データを読み込んでください。")
    st.stop()

# Create Chain with selected sources
if not selected_sources:
    st.warning("検索対象が選択されていません。設定から1つ以上の資料を選択してください。")
    rag_chain = None
else:
    rag_chain = create_rag_chain(vector_store, llm, selected_sources)


# Disclaimer for mobile (fixed position at bottom)
st.markdown("""
    <style>
    .mobile-disclaimer {
        position: fixed;
        bottom: 5px;
        left: 10px;
        font-size: 10px;
        color: #9ca3af; /* Gray-400 */
        z-index: 9999;
        pointer-events: none;
        background-color: rgba(255, 255, 255, 0.7);
        padding: 2px 6px;
        border-radius: 4px;
    }
    </style>
    <div class="mobile-disclaimer">
        ※ AIの回答は間違っている場合もあります (v1.5)
    </div>
    """, unsafe_allow_html=True)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Handle Chat Input
if prompt := st.chat_input("何か質問はありますか？"):
    if not rag_chain:
         st.error("検索対象を選択してください。")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

# 2. Display History
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="assets/new_icon.jpg"):
            # HTMLタグ (<br>等) を有効にするために unsafe_allow_html=True
            st.markdown(message["content"], unsafe_allow_html=True)
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

# 3. Example Questions (Only if history is empty)
if len(st.session_state.messages) == 0:
    # 隙間を詰めるために margin-top をマイナスに指定
    st.markdown("""
        <h3 style='margin-top: -30px; margin-bottom: 10px;'>💡 よくある質問</h3>
        """, unsafe_allow_html=True)
    example_cols1 = st.columns(2)
    example_cols2 = st.columns(2)
    
    examples = [
        "すぎやまって何者なの？",
        "進路について悩んでいる",
        "不登校について悩んでいる",
        "雑談したい"
    ]
    
    # Row 1
    with example_cols1[0]:
        if st.button(examples[0], use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": examples[0]})
            st.rerun()
    with example_cols1[1]:
        if st.button(examples[1], use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": examples[1]})
            st.rerun()
            
    # Row 2
    with example_cols2[0]:
        if st.button(examples[2], use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": examples[2]})
            st.rerun()
    with example_cols2[1]:
        if st.button(examples[3], use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": examples[3]})
            st.rerun()
            
    # スマホ版冒頭画面のスクロール余白 (入力欄と被らないように - スタート時は小さく)
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)

# 4. Generate Response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    if not rag_chain:
        # Should already be handled by chat_input check, but double safety for button clicks
        st.error("検索対象が選択されていません。")
    else:
        with st.chat_message("assistant", avatar="assets/new_icon.jpg"):
            
            # 質問送信直後、自分の質問と「ちょっと待ってね」が見えるようにスクロール
            components.html(
                """
                <script>
                    const scrollToLatest = () => {
                        // モバイルキーボードを閉じる
                        if (window.parent.document.activeElement) {
                            window.parent.document.activeElement.blur();
                        }
                        
                        const messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
                        if (messages.length > 0) {
                            const lastMsg = messages[messages.length - 1];
                            if (lastMsg) lastMsg.scrollIntoView({behavior: 'smooth', block: 'center'});
                        }
                    };
                    setTimeout(scrollToLatest, 100);
                </script>
                """,
                height=0,
            )
            
            with st.spinner("ちょっと待ってね〜"):
                # Spacer AFTER text (inside spinner context doesn't work well visually, 
                # but rendering an empty div here creates space)
                st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                # Convert session state messages to LangChain format
                chat_history = []
                for i in range(0, len(st.session_state.messages) - 1):
                    msg = st.session_state.messages[i]
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))
                
                prompt = st.session_state.messages[-1]["content"]
                response_container = st.empty()
                full_response = ""
                
                try:
                    # 502エラー対策 currently failing because of silence.
                    # Fix: Stream "Thinking..." updates to keep connection alive, 
                    # but accumulate answer for one-shot display.
                    
                    full_response = ""
                    chunk_count = 0
                    loading_texts = ["考え中.", "考え中..", "考え中..."]
                    
                    # Create a placeholder for the "Thinking" animation
                    progress_placeholder = st.empty()
                    
                    for chunk in rag_chain.stream({"input": prompt, "chat_history": chat_history}):
                         if "answer" in chunk:
                             full_response += chunk["answer"]
                             chunk_count += 1

                             # Update "Thinking..." every few chunks to send bytes to client (Keep-Alive)
                             # Don't show the text yet.
                             if chunk_count % 5 == 0:
                                 progress_placeholder.markdown(f"*{loading_texts[chunk_count % 3]}*")

                    # Clear the thinking placeholder
                    progress_placeholder.empty()

                    # ★修正: 先頭の空行・空白を完全に削除 (regex)
                    full_response = re.sub(r'^[\s\n\r]+', '', full_response)

                    # ★後処理フィルター: 「んだ」をOK語尾に強制置換
                    def fix_endings(text):
                        # 「んだ」パターンを「なの」「んだよ」に置換
                        text = re.sub(r'んだ。', 'なの。', text)
                        text = re.sub(r'んだ！', 'んだよ！', text)
                        text = re.sub(r'んだ\n', 'なの\n', text)
                        text = re.sub(r'んだ$', 'なの', text)
                        return text
                    
                    full_response = fix_endings(full_response)

                    # 回答が完成したら一括表示 (Enable HTML for <br>)
                    response_container.markdown(full_response, unsafe_allow_html=True)

                    # 入力欄と被らないように、回答の最後に空行を強制追加
                    full_response += "\n\n<br><br><br>"
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                    # 回答完了後、ユーザーの質問が見える位置までスクロール（回答の先頭付近）
                    # Streamlitのオートスクロール(底への移動)と競合するため、時間差で何度か実行して強制的に位置を合わせる
                    # 回答完了後、ユーザーの質問が見える位置までスクロール（回答の先頭付近）
                    components.html(
                        """
                        <script>
                            const scrollToQuestion = () => {
                                const messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
                                if (messages.length >= 2) {
                                    // 最後のメッセージ(回答)の一つ前(質問)を取得
                                    const questionMsg = messages[messages.length - 2];
                                    if (questionMsg) {
                                        // 質問の上端を画面上端より少し余裕を持って合わせる (block: start)
                                        questionMsg.scrollIntoView({behavior: 'smooth', block: 'start'});
                                    }
                                }
                            };

                            // 複数回実行して適用確率を上げる
                            setTimeout(scrollToQuestion, 100);
                            setTimeout(scrollToQuestion, 500);
                            setTimeout(scrollToQuestion, 1000);
                        </script>
                        """,
                        height=0,
                    )

                except Exception as e:
                    error_msg = f"エラーが発生しました: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": "申し訳ありません。エラーが発生しました。"})

# 5. チャットモード時のみ、最下部に大きな余白を追加（入力欄被り防止）
if len(st.session_state.messages) > 0:
    # PC: 500px, スマホ: 600px 相当のスペーサー -> 1/5以下 (60px)へ変更
    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)


# Validated
