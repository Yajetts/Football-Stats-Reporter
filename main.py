import streamlit as st
import os
from utils.schema import CustomAIAssistant

def main():
    # Set up the Streamlit app title
    st.title("AI Assistant Chatbot")

    # Initialize session state for the assistant and chat history
    if "assistant" not in st.session_state:
        st.session_state.assistant = None
        st.session_state.messages = []

    # Initialize or load the AI assistant
    if st.session_state.assistant is None:
        data_path = "./data"  # Directory containing your documents
        index_path = "./index"  # Directory to store the vector index
        
        try:
            st.session_state.assistant = CustomAIAssistant(
                data_path=data_path,
                index_path=index_path
            )
            status_msg = "Loaded existing index." if os.path.exists(index_path) else "Created new index."
            st.success(status_msg)
        except Exception as e:
            st.error(f"Error initializing assistant: {str(e)}")
            return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input and response generation
    if prompt := st.chat_input("Ask your question here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                result = st.session_state.assistant.query(prompt)
                st.markdown(result.answer)
                
                # Optional: Display sources if available
                if result.source_nodes:
                    with st.expander("Sources"):
                        for source in result.source_nodes:
                            st.write(source)
                
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": result.answer}
                )
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )

if __name__ == "__main__":
    main()