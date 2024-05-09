
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
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
import os
import time

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Access the loaded environment variables
api_key = os.getenv("GEMINI_API_KEY")


@st.cache_resource()
def load_model(model_name, tokenizer_name):
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_name)
    return model, tokenizer


model, tokenizer = load_model(
    "facebook/mbart-large-50-many-to-many-mmt", "facebook/mbart-large-50-many-to-many-mmt")

# st.set_page_config(page_title="Document Genie", layout="wide")


contents = {
    'english': """
# AIRWR Project
## 🚀 Meet Chinku - Your Knowledge Buddy 🌟

Feeling lost with news or PDFs? Don't worry, Chinku is here to help you understand everything easily.

Chinku loves making complex things simple. Whether it's news stories or tricky documents, Chinku will guide you through it all.

Relax, and let Chinku simplify your reading journey. Get ready to explore the world with ease! 🚀📚
""",
    'hindi': """
# AIRWR प्रोजेक्ट
## 🚀 मिलें चिंकू - आपका ज्ञान साथी 🌟

समाचार या पीडीएफ में खो गए हैं? चिंकू आपको सब कुछ आसानी से समझने में मदद करने के लिए यहाँ है।

चिंकू को जटिल चीजों को सरल बनाने का बहुत शौक है। चाहे वह समाचार कहानियाँ हों या कठिन दस्तावेज़, चिंकू आपको सब कुछ समझाने में मदद करेगा।

आराम करें, और चिंकू को अपने पढ़ाई की यात्रा को सरल बनाने की अनुमति दें। आसानी से दुनिया का अन्वेषण करने के लिए तैयार हो जाएं! 🚀📚
""",
    'malayalam': """
# ഏയർഡബ്ല്യൂആർ പ്രൊജക്ട്ട്
## 🚀 ചിങ്കു - നിങ്ങളുടെ അറിവ് സുഹൃത്ത് 🌟

വാർത്തകൾ അല്ലെങ്കിൽ പിഡിഎഫുകൾക്ക് വെള്ളപ്പെട്ടത് തിരഞ്ഞെടുക്കുന്നില്ലെങ്കിൽ? ചിങ്കു എല്ലാം സുലഭമായി പഠിക്കാൻ നിങ്ങൾക്ക് സഹായിക്കുന്നതിന് ഇവിടെയാണ്.

ചിങ്കു ഉള്ളങ്കളൈ എള്ളതാക്കുന്നത് ഇഷ്ടമാണ്. സമാചാര കഥകൾ അല്ലെങ്കിൽ കഠിനമായ രേഖകൾ ആയിരിക്കാം, ചിങ്കു അവയെ എല്ലാം ഉങ്ങളുക്ക് എളുപ്പത്തിൽ അറിയിക്കും.

ചിലൈയൈപ്പുറമ്പുറമായി ചിങ്കുവിന്റെ നിങ്ങളുടെ വായനാ യാത്ര സുവിശേഷമായി സുലഭമാക്കാൻ തയ്യാറാകുക! 🚀📚
""",
    'telugu': """
# AIRWR ప్రాజెక్ట్
## 🚀 చింకు కనీసం పరిజ్ఞాన బుడ్డి 🌟

వార్తలు లేదా PDF ఫైళ్లలో హారైపోయారా? చింకు ఇప్పుడు మీరు అన్నింటినీ సులభంగా అర్థం చేసుకుంటారు.

చింకు కఠినమైన విషయాలను సరళంగా చేసేందుకు అతడు ప్రియతమ. సమాచార కథలంటే లేదా కఠినమైన దస్తావేజులంటే, చింకు మీరు అన్నింటినీ సరళంగా అర్థం చేసేందుకు మద్దతు చేస్తుంది.

ఉలికెప్పుడు, చింకుకు మీ చదవడానికి చిన్న ప్రయాసంతో ఆరామం చేయండి. సులభంగా ప్రపంచాన్ని అన్వేషించడానికి సిద్ధంగా ఉండండి! 🚀📚
""",
    'tamil': """
# ஏய்ஆர்டப்ள்யூஆர் திட்டம்
## 🚀 சிங்கு - உங்கள் அறிவு சகோதரி 🌟

செய்திகளை அல்லெங்கி பி.டி.எஃப்களை படிக்கும் போது கூட பொருட்களை காட்டும் நினைவு உள்ளதா? சிங்கு அதை எளிதாக உங்களுக்கு புரிந்துகொள்ள உதவுகின்றன.

சிங்கு ஜடிலமாய் கார்யங்களை எளிதாக்குவது இஷ்டம். அதனால், செய்தி கதைகள் அல்லெங்கி கடினமான ஆவணங்கள் ஆகியவற்றையும் சிங்கு உங்களுக்கு வழங்கும்.

விரம்பிச் சிங்கு உங்கள் வாயின் பயணத்தை எளிதாக்கும். உலகத்தை ஏற்றுக்கொள்ள உதவுகிறார்! 🚀📚
"""
}


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_url_content(url: str):

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
        chunk_size=1000, chunk_overlap=500)
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


def user_input(user_question, api_key, src):
    """embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local(
        "faiss_index", user_question, embedding=embeddings, allow_dxangerous_deserialization=True)"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key)

    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question, fetch_k=5)
    print("+++"*50)
    print("Similar docs", docs)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)
    text = response["output_text"]
    print(text)
    st.write("Reply: ", translate_en_ln(text, src))


def translate_ln_en(article_hi, src):
    tokenizer.src_lang = src

    encoded_hi = tokenizer(article_hi, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return text


def translate_en_ln(article_en, trg):
    tokenizer.src_lang = "en_XX"
    encoded_en = tokenizer(article_en, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id[trg])
    text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return text[0]


def main():
    languages = ['english', 'hindi', 'malayalam', 'tamil', 'telugu']

    st.markdown(contents[languages[0]])

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
        # user_question = st.text_input("Ask a Question", key="user_question")

    elif option == "Enter URLs":
        st.header("Enter URLs")
        urls = st.text_area("Enter multiple URL")
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
    # ['english', 'hindi', 'malayalam', 'tamil','telugu']

    option = st.selectbox(
        'Language/భాష/भाषा/மொழி/ഭാഷ',
        ('Hindi/हिंदी', 'Telugu/తెలుగు',  'Tamil/தமிழ்', 'malayalam/ഭാഷ'))
    if option == 'Telugu/తెలుగు':
        # st.markdown(contents[languages[4]])
        src = "te_IN"
    elif option == 'Hindi/हिंदी':
        # st.markdown(contents[languages[1]])
        src = "hi_IN"
    elif option == 'Tamil/தமிழ்':
        # st.markdown(contents[languages[2]])
        src = "ta_IN"
    elif option == 'malayalam/ഭാഷ':
        # st.markdown(contents[languages[3]])
        src = "ml_IN"
    user_question = st.text_input(
        "Ask a Question", key="user_question")
    if st.button("Submit") and api_key:
        user_question = translate_ln_en(user_question, src)
        if user_question and api_key:
            user_input(user_question, api_key, src)


if __name__ == "__main__":
    main()
