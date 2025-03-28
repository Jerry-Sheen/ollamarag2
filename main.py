import streamlit as st
from langchain_community.llms import Ollama
import os
from langchain_community.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from my_modules import modelName
from dotenv import load_dotenv
import requests

# from langchain_sidebar_content import LC_QuickStart_01
# from langchain_ollama import OllamaLLM

#llm = Ollama(model="gemma:7b", temperature=0)
#response = llm.invoke("아재 개그를 해줘")
#print(response)

# 환경 변수 설정
load_dotenv()
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')  # 기본값 설정

def connect_ollama():
    """Ollama 서버 연결 테스트"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        return response.status_code == 200
    except:
        return False

def generate_text(input_text, whatToAsk, language):
    try:
        if not connect_ollama():
            raise ConnectionError("Ollama 서버 연결 실패")
            
        # 나머지 코드는 동일
        # ...
        
    except Exception as e:
        st.error(f"연결 오류: {str(e)}")
        st.markdown("""
        **해결 방법 체크리스트**
        1. 원격 서버에서 Ollama 실행 확인
        2. 방화벽에서 11434 포트 개방
        3. `OLLAMA_HOST` 환경변수 설정 확인
        """)

def createChain(llm, output_parser):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a caring teacher who answers students' questions."),
    ("user", "{input}")
    ])
    if output_parser:
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser   
    else:
        chain = prompt | llm  

    return chain


def generate_text(input_text, whatToAsk, language):
    try: 
        model_name = modelName()
         # Initialize your OpenAI instance using the provided API key
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = Ollama(model=model_name, callbacks=callback_manager)
        if (whatToAsk == 'Basic'):
            st.write("- *Simpleast way to use LLM.*")
            generated_text = llm.invoke("Please answer for this question. in " + language + ". " + input_text )
        elif (whatToAsk == 'ChatPromptTemplate'):
            output_parser = False
            chain = createChain(llm, output_parser)
            st.write("- *Prompt templates are used to convert raw user input to a better input to the LLM.*")
            generated_text = chain.invoke({"input": "Please answer for this question. in " + language + ". " + input_text})
        else:
            output_parser = True
            chain = createChain(llm, output_parser)
            generated_text = chain.invoke({"input": "Please answer for this question. in " + language + ". " + input_text})
        return generated_text
    except Exception as e:
        st.warning("could not connect to ollama")
        st.warning(e)


def main():
    st.title('KRIBB AI Network2')

    # Get user input for topic of the poem
    input_text = st.text_input('Throw a question, please!')

    whatToAsk = st.radio(
    "Please choose one way to ask an LLM question from the list below.",
    ["Basic", "ChatPromptTemplate", "StrOutputParser"],
    captions = ["Simplest way.", "Use ChatPromptTemplate with chain.", "Add StrOutputParser to the chain"])

    # List of languages available for ChatGPT
    available_languages = ["English", "Korean", "Spanish", "French", "German", "Chinese", "Japanese"]

    # User-selected language
    selected_language = st.selectbox("Select a language:", available_languages)  

    # Button to trigger text generation
    if st.button("Submit."):
        with st.spinner('Wait for it...'):
            # When an API key is provided, display the generated text
            generated_text = generate_text(input_text, whatToAsk, selected_language)
            st.write(generated_text)

     
if __name__ == "__main__":
    main()
st
#LC_QuickStart_01()


