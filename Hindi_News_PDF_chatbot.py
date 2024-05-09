
# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from typing import Optional
import requests
from bs4 import BeautifulSoup
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import tensorflow as tf
import time
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Access the loaded environment variables
api_key = os.getenv("GEMINI_API_KEY")


@st.cache_resource()
def load_model_1(model_name, tokenizer_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, from_tf=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


@st.cache_resource()
def load_model_2(model_name, tokenizer_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, from_tf=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


model1, tokenizer1 = load_model_1(
    "Nettem-Gayathri/HI-EN-translation", "Nettem-Gayathri/HI-EN-translation")

model2, tokenizer2 = load_model_2(
    "Nettem-Gayathri/EN-HI-translation", "Nettem-Gayathri/EN-HI-translation")


st.markdown("""
# AIRWR प्रोजेक्ट
## 🚀 मिलें चिंकू - आपका ज्ञान साथी 🌟

समाचार या पीडीएफ में खो गए हैं? चिंकू आपको सब कुछ आसानी से समझने में मदद करने के लिए यहाँ है।

चिंकू को जटिल चीजों को सरल बनाने का बहुत शौक है। चाहे वह समाचार कहानियाँ हों या कठिन दस्तावेज़, चिंकू आपको सब कुछ समझाने में मदद करेगा।

आराम करें, और चिंकू को अपने पढ़ाई की यात्रा को सरल बनाने की अनुमति दें। आसानी से दुनिया का अन्वेषण करने के लिए तैयार हो जाएं! 🚀📚
""")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_url_content(url: str):
    """Fetches the text content from a URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        target_classes = ["article__content-container",
                          "sp-cn ins_storybody", "styles_article__Ee5Ad", "article", "col-xl-9 col-lg-8 col-md-12 col-sm-12 col-12 storyline", "_s30J clearfix", "storyDetails", "story_details", "fullStory tfStory current videoStory story__detail storyAf101713237848782"]
        for class_name in target_classes:
            div_elements = soup.find_all("div", class_=class_name)
            for div in div_elements:
                text_content = div.get_text()
                print(f"Text content from {url}:")
                print(text_content.strip())
                print("=" * 50)
                return (text_content.strip())
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")

        """text = ""
        response = requests.get(url)
        response.raise_for_status()
        text += response.text
        print(text)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from URL: {url}")
        print(e)
        return None"""


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, api_key):
    """embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key)
    # vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store = FAISS.load_local(
        "faiss_index", text_chunks, embedding=embeddings, allow_dangerous_deserialization=True)"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(
        text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context but here is the summary of the file you provided" and print the summary, don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, api_key):
    """embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local(
        "faiss_index", user_question, embedding=embeddings, allow_dangerous_deserialization=True)"""
    user_question = translate_hi_en(user_question)
    print(user_question)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key)

    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    print("+++"*50)
    print("Similar docs", docs)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", translate_en_hi(response["output_text"]))


def translate_hi_en(article_hi):
    """tokenizer.src_lang = "hi_IN"

    encoded_hi = tokenizer(article_hi, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return text"""
    inputs = tokenizer1(article_hi, return_tensors="pt")
    outputs = model1.generate(**inputs)
    translated_text = tokenizer1.decode(outputs[0], skip_special_tokens=True)
    print("Translated Text:", translated_text)
    return translated_text


def translate_en_hi(article_en):
    sentences = article_en.split('.')
    translated_sentences = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        inputs = tokenizer2(sentence, return_tensors="pt")
        outputs = model2.generate(**inputs)
        translated_sentence = tokenizer2.decode(
            outputs[0], skip_special_tokens=True)
        translated_sentences.append(translated_sentence)
    translated_text = ".".join(translated_sentences)
    print("Translated Text:", translated_text)
    return translated_text


def main():
    st.sidebar.title("Menu:")
    option = st.sidebar.selectbox(
        "Select Option:", ["Enter URLs", "Upload PDFs"])

    if option == "Upload PDFs":
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDF Files", accept_multiple_files=True)
        if st.button("Process PDFs") and api_key:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                print("raw_text", raw_text)
                print("==="*50)
                print("raw_text type ", type(raw_text))
                text_chunks = get_text_chunks(raw_text)
                print("text_chunks", text_chunks)
                print("==="*50)
                print("text_chunks type ", type(text_chunks))
                get_vector_store(text_chunks, api_key)
                st.success("PDFs Processed")
        user_question = st.text_input("Ask a Question", key="user_question")

    elif option == "Enter URLs":
        st.header("Enter URLs")
        urls = st.text_area("Enter URLs (one per line):")
        if st.button("Process URLs") and api_key:
            with st.spinner("Processing URLs..."):

                url_contents = []
                for url in urls.splitlines():
                    url_content = get_url_content(url)
                    url_contents.append(url_content)

                all_text_chunks = []
                for url_content in url_contents:
                    text_chunks = get_text_chunks(url_content)
                    all_text_chunks.extend(text_chunks)

                get_vector_store(all_text_chunks, api_key)
                st.success("URLs Processed")
        user_question = st.text_input(
            "Ask a Question", key="user_question")

    if st.button("Submit") and user_question and api_key:
        user_input(user_question, api_key)


if __name__ == "__main__":
    main()
