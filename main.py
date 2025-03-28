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

# Ollama 호스트 설정
os.environ['OLLAMA_HOST'] = 'http://localhost:11434'

load_dotenv()  # .env 파일에서 환경 변수 로드

# Ollama API 엔드포인트
OLLAMA_URL = f"{os.environ['OLLAMA_HOST']}/api/generate"

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
    st.title('KRIBB AI Network')

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


