# AIすぎやま 公開手順ガイド

このアプリをインターネット上で一般公開するための手順です。
**「Streamlit Community Cloud」** という無料サービスを使用します。

## 手順1: GitHubへのアップロード

まず、このプロジェクトをGitHubというコード管理サイトにアップロードする必要があります。

1.  **GitHubアカウント**をお持ちでない場合は、[GitHub.com](https://github.com/)で作成してください。
2.  GitHub上で **「New repository」** をクリックし、新しいリポジトリ（箱）を作成します。
    *   Repository name: `ai-sugiyama` （など好きな名前）
    *   Public（公開）を選択
    *   「Create repository」をクリック
3.  ターミナルで以下のコマンドを実行し、作成したリポジトリにコードをアップロードします。
    *   （※ `YOUR_GITHUB_USERNAME` は自分のユーザー名に置き換えてください）

```bash
# 1. Gitの初期化
git init

# 2. 全ファイルをステージング（.gitignoreの設定により、.envなどは除外されます）
git add .

# 3. コミット（保存）
git commit -m "Initial commit"

# 4. リモートリポジトリの登録（GitHubの画面にあるURLを使ってください）
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/ai-sugiyama.git

# 5. アップロード
git push -u origin main
```

## 手順2: Streamlit Community Cloudでの公開

1.  [Streamlit Community Cloud](https://share.streamlit.io/) にアクセスし、サインインします（GitHubアカウントでログインできます）。
2.  **「New app」** をクリックします。
3.  **「Use existing repo」** を選択し、先ほど作成したリポジトリ（`YOUR_GITHUB_USERNAME/ai-sugiyama`）を選択します。
4.  設定項目を確認します：
    *   **Repository:** `ai-sugiyama`
    *   **Branch:** `main`
    *   **Main file path:** `app.py`
5.  **「Advanced settings」** をクリックし、**Secrets** を設定します。
    *   ここにAPIキーを保存することで、安全にキーを管理できます。
    *   以下の内容をコピー＆ペーストし、`あなたのAPIキー` の部分を実際のキー（`.env`ファイルの中身と同じもの）に書き換えてください。

```toml
GOOGLE_API_KEY = "AIzaSy..."
```

6.  **「Save」** を押し、最後に **「Deploy!」** をクリックします。

## 完了！

数分待つと、アプリが起動します。
発行されたURL（例: `https://ai-sugiyama.streamlit.app`）をSNSなどでシェアすれば、誰でも「AIすぎやま」と会話できるようになります！
