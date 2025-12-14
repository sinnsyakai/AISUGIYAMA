import streamlit as st

st.set_page_config(page_title="AI Sugiyama Test", page_icon="🧪")

st.title("🧪 AI Sugiyama テスト")
st.write("アプリが正常に動作しています！")

# Basic test
if st.button("テストボタン"):
    st.success("ボタンが動作しています！")
