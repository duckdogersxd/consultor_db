import streamlit as st
import os
import re
import time
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory

# --- Configuração da Página (Melhoria #1: Interface Web) ---
st.set_page_config(page_title="Consultor de Dados IA", layout="wide")

# --- Barra Lateral para Configuração ---
with st.sidebar:
    st.header("🔐 Configuração")
    api_key = st.text_input("Insira sua Google API Key", type="password", help="Pegue sua chave no Google AI Studio")
    st.markdown("[Obter chave aqui](https://aistudio.google.com/app/apikey)")
    st.divider()
    st.info("A chave é usada apenas nesta sessão e não fica salva.")
    
# --- Cache de Recursos (Para não recarregar o banco a cada clique) ---
@st.cache_resource
def load_db_assistant():
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("ERRO: Configure a GOOGLE_API_KEY no arquivo .env")
        return None

    # Verifica se o banco existe
    if not os.path.exists("./chroma_db"):
        st.error("ERRO: Banco de dados não encontrado. Rode 'python ingest.py' primeiro.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Modelo Gemini 1.5 Flash (Estável e com boa cota)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)
    
    return {"llm": llm, "retriever": vectorstore.as_retriever()}

# --- Lógica do Chat (Backend) ---
def get_response(user_input, resources, chat_history):
    llm = resources["llm"]
    retriever = resources["retriever"]

    # 1. Recupera contexto (RAG)
    docs = retriever.invoke(user_input)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # 2. Prompt com Auto-Correção (Versão Blindada)
    # Mudança: Removemos o 'f' do início e passamos {context} como variável do LangChain
    system_prompt = """
    Você é um Arquiteto de Dados Sênior e Consultor.
    
    CONTEXTO TÉCNICO (TCC):
    {context}

    SEU OBJETIVO:
    Organizar as informações do usuário em um modelo de dados claro.
    
    REGRAS DE GERAÇÃO DE DIAGRAMA (#5 AUTO-CORREÇÃO):
    - Se for gerar um diagrama, use APENAS a sintaxe `erDiagram` do Mermaid.
    - NÃO use espaços em nomes de entidades (use `snake_case`).
    - NÃO use caracteres especiais ou acentos dentro das chaves {{}}. 
    - Defina PK e FK explicitamente.
    - REVISE O CÓDIGO MERMAID ANTES DE RESPONDER. Garanta que todas as chaves abertas {{ foram fechadas }}.

    FORMATO:
    - Explique o raciocínio em português.
    - Se houver diagrama, coloque-o dentro de blocos ```mermaid ... ```.
    """

    # Monta a chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()
    
    # 3. Execução
    # Aqui passamos o "context" explicitamente no dicionário
    return chain.invoke({
        "input": user_input,
        "history": chat_history,
        "context": context_text
    })

# --- Função para Renderizar Mermaid (Melhoria #1 e #2) ---
def render_mermaid(code):
    """
    Injeta Javascript para renderizar o diagrama no navegador.
    Isso permite visualizar o diagrama sem instalar nada no PC.
    """
    html_code = f"""
    <div class="mermaid">
    {code}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    """
    return components.html(html_code, height=500, scrolling=True)

# --- Interface Principal ---
def main():
    st.title("🤖 Consultor de Modelagem de Dados")
    st.caption("Assistente Inteligente com RAG + Gemini 1.5")

    # Inicializa sessão
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Carrega o "cérebro"
    resources = load_db_assistant()
    if not resources:
        return

    # Layout: Coluna 1 (Chat) | Coluna 2 (Diagrama Interativo)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("💬 Chat")
        # Exibe histórico
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input do usuário
        if prompt := st.chat_input("Descreva seu negócio ou peça uma alteração..."):
            # 1. Mostra pergunta
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Processa resposta
            with st.chat_message("assistant"):
                with st.spinner("Analisando contexto e regras do TCC..."):
                    # Converte histórico para formato do LangChain
                    history_langchain = []
                    for m in st.session_state.messages:
                        if m["role"] == "user":
                            from langchain_core.messages import HumanMessage
                            history_langchain.append(HumanMessage(content=m["content"]))
                        else:
                            from langchain_core.messages import AIMessage
                            history_langchain.append(AIMessage(content=m["content"]))

                    response = get_response(prompt, resources, history_langchain)
                    
                    # Remove o bloco mermaid do texto para não duplicar na tela
                    text_display = re.sub(r"```mermaid\n(.*?)\n```", "", response, flags=re.DOTALL)
                    st.markdown(text_display)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun() # Atualiza a tela para renderizar o diagrama na coluna ao lado

    with col2:
        st.subheader("🖼️ Diagrama em Tempo Real")
        
        # Procura o último diagrama gerado no histórico
        last_mermaid_code = ""
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant":
                match = re.search(r"```mermaid\n(.*?)\n```", msg["content"], re.DOTALL)
                if match:
                    last_mermaid_code = match.group(1)
                    break
        
        if last_mermaid_code:
            # --- Melhoria #2: Edição Interativa ---
            st.info("Você pode editar o código abaixo para corrigir detalhes manualmente:")
            
            # Caixa de texto editável com o código gerado
            edited_code = st.text_area(
                "Código Mermaid (Editável)", 
                value=last_mermaid_code, 
                height=200,
                key="mermaid_editor"
            )
            
            st.write("---")
            st.write("**Visualização:**")
            # Renderiza o código (original ou editado pelo usuário)
            render_mermaid(edited_code)
            
            # Botão para baixar (Extra de usabilidade)
            st.download_button(
                label="📥 Baixar Código do Diagrama",
                data=edited_code,
                file_name="diagrama.mmd",
                mime="text/plain"
            )
        else:
            st.info("Descreva seu cenário no chat para gerar um diagrama aqui.")

if __name__ == "__main__":
    main()