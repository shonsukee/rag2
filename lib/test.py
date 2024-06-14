import streamlit as st
import time

# チャットメッセージを格納するリストをセッション状態に初期化
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ボタン追加
def click_button(idx):
    st.session_state.messages[idx]["flag"] = True
    st.rerun()

# メッセージとボタンの表示
for i, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            if not message["flag"]:
                if st.button('👍', key=f"button_{i}"):
                    with st.spinner('inserting...'):
                        time.sleep(3)
                    click_button(i)
            else:
                st.write("データを追加しました")

# 新しいメッセージの入力
if prompt := st.chat_input("Please enter what you want to search for:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    key_cnt = len(st.session_state["messages"])
    st.session_state.messages.append({"role": "assistant", "content": prompt, "flag": False})
    with st.chat_message("assistant"):
        st.write(prompt)
        if not st.session_state.messages[-1]["flag"]:
            if st.button('🙌', key=f"button_{key_cnt}", on_click=click_button, args=(key_cnt,)):
                click_button(key_cnt)
        else:
            st.write("データを追加しました")
