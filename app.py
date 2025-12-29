import streamlit as st
import os
import langchain 
st.write("LangChain version:", langchain.__version__)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# --- 1. CONFIGURAÇÃO DE SEGURANÇA ---
# No Streamlit Cloud, coloque a chave em Settings -> Secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("⚠️ ERRO: Configure a 'GOOGLE_API_KEY' nos Secrets do Streamlit.")
    st.stop()

# --- 2. MOTOR DE INTELIGÊNCIA (RAG) ---
@st.cache_resource
def carregar_base_conhecimento():
    """Lê os arquivos na pasta 'legislacao' e cria o banco de dados da IA."""
    if not os.path.exists("legislacao"):
        os.makedirs("legislacao")
        
    # Carrega PDFs e DOCX
    pdf_loader = DirectoryLoader('legislacao/', glob="./*.pdf", loader_cls=PyPDFLoader)
    docx_loader = DirectoryLoader('legislacao/', glob="./*.docx", loader_cls=Docx2txtLoader)
    
    docs = pdf_loader.load() + docx_loader.load()
    
    if not docs:
        return None

    # Divide o texto em blocos para a IA não se perder
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        separators=["\nArt.", "\n§", "\n\n", "\n"]
    )
    chunks = text_splitter.split_documents(docs)

    # Cria o banco de vetores (Embeddings do Gemini)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# --- 3. DEFINIÇÃO DO SYSTEM PROMPT (O "CÉREBRO" DO CONSULTOR) ---
system_template = """
Você é o Consultor Legislativo Sênior da Câmara Municipal de Espumoso, RS.
Sua missão é fornecer análises técnicas fundamentadas na Lei Orgânica, Regimento Interno e Regime Jurídico.

DIRETRIZES:
1. LEGALIDADE: Use linguagem formal e cite sempre o Artigo, Parágrafo ou Inciso (ex: Art. 12, §1º).
2. CONTEXTO: Utilize APENAS os documentos fornecidos para responder. 
3. SE NÃO SOUBER: Se a lei não mencionar o assunto, diga: "Não localizei previsão específica na legislação disponível."
4. ESTILO: Seja pragmático, direto e técnico.

CONTEXTO DOS DOCUMENTOS:
{context}

HISTÓRICO DA CONVERSA:
{chat_history}
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

# --- 4. INTERFACE DO USUÁRIO (STREAMLIT) ---
st.set_page_config(page_title="IA Legislativa Espumoso", page_icon="🏛️")
st.title("🏛️ Consultor Legislativo Digital")
st.subheader("Câmara Municipal de Espumoso/RS")

# Inicializa o Banco de Dados
vector_db = carregar_base_conhecimento()

if vector_db is None:
    st.info("📌 Por favor, adicione os arquivos PDF na pasta 'legislacao' para começar.")
    st.stop()

# Inicializa Memória e Chain
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key='answer'
    )

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
    memory=st.session_state.memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    return_source_documents=True
)

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Em que posso ajudar hoje?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando legislação..."):
            response = qa_chain.invoke({"question": prompt})
            answer = response['answer']
            
            # Extrai as fontes para transparência
            sources = set([os.path.basename(doc.metadata['source']) for doc in response['source_documents']])
            source_text = f"\n\n---\n**Fontes:** {', '.join(sources)}"
            
            full_response = answer + source_text
            st.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

