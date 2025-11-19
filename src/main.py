import streamlit as st

from agent import ui_agent # pyright: ignore[reportUnknownVariableType]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in ui_agent.stream( # pyright: ignore[reportUnknownMemberType]
            {"messages": [{"role": "user", "content": prompt}]},
            stream_mode="updates",
        ):
            for step, data in chunk.items():
                content = data['messages'][-1].content_blocks[-1]
                if content['type'] == 'text':
                    full_response += content['text'] + " "
                    message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
