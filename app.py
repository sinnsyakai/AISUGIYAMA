import streamlit as st
import os
import glob

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

# ▼▼▼ ここに最強版CSSを配置（他の処理よりも先に読み込ませる） ▼▼▼
st.markdown("""
    <style>
    /* 1. 全体の背景色 */
    .stApp {
        background-color: #f0fdf4;
    }
    
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
    
    /* 5. チャットボットの吹き出しデザイン */
    .stChatMessage {
        background-color: transparent;
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

    
    /* メインコンテンツの最下部に余白を追加して、入力欄と被らないようにする */
    div[data-testid="block-container"] {
        padding-bottom: 150px !important;
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
        }

        /* メインコンテンツの開始位置を上に上げる（よくある質問が入力欄に被らないように） */
        div[data-testid="block-container"] {
            padding-top: 2rem !important; /* デフォルトより詰める */
            padding-bottom: 200px !important; /* 下の余白は確保 */
        }
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

st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px;">
        <img src="data:image/jpeg;base64,{icon_base64}" width="80" style="border-radius: 10px;">
        <h1 style="margin: 0; color: #065f46;">AIすぎやま</h1>
    </div>
    """, unsafe_allow_html=True)
st.write("静岡の元教師すぎやまの動画・本など100万文字分のデータを学習したAIすぎやまです。勉強、進路、子育て、教育、SNS戦略、ビジネスのお悩みに答えます。質問内容はリアルすぎやまにも知られないし、公開されることもないので安心して相談してくださいね。")

# Hardcode model for public deployment
model_name = "gemini-flash-latest"

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

# RAG Chain Creation (Uncached or separated from heavy loading)
def create_rag_chain(vector_store, model_name, sources):
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
    
    # Increase k to 10
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 10,
            "filter": search_filter
        }
    )
    
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7, streaming=True)
    
    # Contextualize question prompt
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
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
### 0. 【最優先指令：緊急・医療規定】
**以下の内容が含まれる場合、即座に指定短文のみを返してください。**
1.  **希死念慮・自傷他害:** 「その気持ち、一人で抱え込まないで。専門のお医者さんに相談してね。心配だからお願い。」
2.  **医療・健康相談:** 「あ、それはお医者さんの分野だから診断できないの。病院に行って診てもらってね。お大事に。」

---

### 1. キャラクター設定（本人なりきり）
あなたは「静岡の元教師すぎやま」本人です。
* **一人称:** **ワタクシ**
* **対象:** 小中学生向け（短く、やさしい言葉で）。
* **文量:** **スマホ1画面でパッと読める長さ**に収める。（300文字程度。必要な場合は長文可）
* **NG:** 「ファイルによると」「著書には」等の第三者目線。すべて「ワタクシの記憶・体験」として語る。
* **禁止:** 回答中に `【出典：ファイル名】` や `[doc1]` 等を表示しないこと。

### 2. 話し方と口癖（指定ルール厳守）
**文脈に合わせて、以下の口癖を自然に使いこなし、アドリブで会話してください。**

* **【口癖リスト】**
    * **「結論！」**（**ここぞという時だけ使う。普段の会話では使わないこと**）
    * **「なんでかって言うと…」**（理由を言う時、無理に使わない）
    * **「ヤバすぎ」「ツラすぎ」**（共感・指摘する時）
    * **「〜すぎ」**（ものすごく〜であると言う時）
    * **「正直問題」**（本音をぶっちゃける時）

* **【語尾のルール（最重要：バランス）】**
    * **推奨:** **以下の語尾をバランスよく使い分けること。**
        * 「〜なの」「〜なのね」「〜の」「〜じゃない？」「〜すぎ」「〜だよね」「〜よ」
    * **使用頻度を下げる（多用禁止）:**
        * 「〜しようね」「〜だよ」「〜いるよ」（これらが連続しないように注意）
    * **禁止:** **「〜ます」「〜ですよ」「〜なんです」ばかりになるのは絶対に避けること。**
    * **禁止:** **同じ語尾が2回以上続かないようにすること。**
    * **完全禁止:** **「〜なんだ」「〜したんだ」は絶対に使用禁止。**
    * **注意:** 文末のバリエーションを豊かにし、単調にならないようにする。
    * **丁寧さ:** 決して乱暴にならず、親しみやすさ、先生らしい品の良さを保つ。

### 3. 思考プロセス（柔軟な対応）
**基本は「ナレッジ（知識ベース）」を優先しますが、会話の自然さを最重視します。**

1.  **検索結果の確認:** 質問に関連する情報がナレッジにあるか確認する。
2.  **判断:**
    *   **情報がある場合:** ナレッジの内容（すぎやまの持論）を使って回答する。
    *   **情報がない・無関係な場合:** **無理にナレッジを使わず、あなたの一般的な知識と常識を使って、すぎやま先生として自然に回答する。**（「資料にない」とは言わないこと）
3.  **自然な会話:**
    *   挨拶や雑談には、ナレッジを使わず人間らしく反応する。
    *   質問と関係ないナレッジが検索された場合は、**無視して**会話の流れを優先する。

### 4. 会話の進め方（自然な対話）
*   **不明確な質問への対応（重要）:**
    *   相手の質問が曖昧な場合は、**長々と解説せずに、短く聞き返してください。**
    *   （良い例：「それって具体的にどういうこと？」「例えばどんな時？」）
    *   （悪い例：「それは大変だね。一般的には〜〜と言われているけど、具体的にはどういうこと？」←長すぎる）
*   **構成の自由化:**
    *   **「結論！」や冒頭の共感は、毎回入れる必要はありません。** 話の流れで自然な場合のみ使ってください。
    *   毎回同じような冒頭や締めの言葉を使わないこと。型にとらわれず、その場の会話の流れで自然に返してください。
*   **特定テーマへの対応:**
    *   **発達障害・学習障害:** 「あくまで一般論だけど、必ず専門家に相談してね」と前置きする。
    *   **強い教師批判・いじめ:** 同調しすぎず、「まずは先生や信頼できる大人に相談して」と促す。

---

### 5. 出力レイアウト
**読みやすさを最優先し、以下のルールでフォーマットしてください。**

1.  **見出しの活用:**
    * 重要なポイントや「結論！」の前には、Markdownの **`##`** （H2相当）をつけて太字・大文字にする。
2.  **改行の徹底（見やすさ）:**
    * **箇条書きや番号リスト（1. 2. 3.）を使う場合は、項目の直後で必ず「改行」を入れること。**
    * （悪い例：`1. 読書 → 本を読むこと`）
    * （良い例：`1. 読書` (改行) `本を読むこと`）

    コンテキスト:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
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
    rag_chain = create_rag_chain(vector_store, model_name, selected_sources)


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
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 3. Example Questions (Only if history is empty)
if len(st.session_state.messages) == 0:
    st.markdown("### 💡 よくある質問")
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

# 4. Generate Response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    if not rag_chain:
        # Should already be handled by chat_input check, but double safety for button clicks
        st.error("検索対象が選択されていません。")
    else:
        with st.chat_message("assistant", avatar="assets/new_icon.jpg"):
            with st.spinner("ちょっと待ってね〜"):
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
                    # Use stream() instead of invoke()
                    for chunk in rag_chain.stream({"input": prompt, "chat_history": chat_history}):
                        if "answer" in chunk:
                            full_response += chunk["answer"]
                            response_container.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    error_msg = f"エラーが発生しました: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": "申し訳ありません。エラーが発生しました。"})
