import streamlit as st

# チャットメッセージを格納するリストをセッション状態に初期化
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# チャットメッセージの表示
for message in st.session_state.messages:
    if message["role"] == "assistant":
        st.write(f"Assistant: {message['content']}")
        with st.expander(message["expandar_title"]):
            st.write(message["expandar_content"])
    else:
        st.write(f"User: {message['content']}")

# 新しいメッセージの入力
new_message = st.text_input("Enter your message:", key="new_message")

# メッセージの送信ボタン
if st.button("Send"):
    if new_message:
        st.session_state.messages.append({"role": "user", "content": new_message})
        # ここでアシスタントの応答を追加
        st.session_state.messages.append({"role": "assistant", "content": new_message, "expandar_title": "title pl","expandar_content": "ok content"})
        # 入力フィールドをクリア
        st.experimental_rerun()
