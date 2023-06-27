from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
import pinecone
#importar funcion de otra directorio templates

from templates.condense_promp import CONDENSE_PROMPT
from templates.qa_promp import QA_PROMPT

# from langchain.vectorstores import Chroma


def query(openai_api_key, pinecone_api_key, pinecone_environment, pinecone_index, pinecone_namespace, temperature, sources, use_pinecone):
    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002', openai_api_key=openai_api_key)

    if use_pinecone:
        pinecone.init(api_key=pinecone_api_key,
                      environment=pinecone_environment)
        vectorstore = Pinecone.from_existing_index(
            index_name=pinecone_index, embedding=embeddings, text_key='text',  namespace=pinecone_namespace)

    model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=temperature, 
                       openai_api_key=openai_api_key, streaming=True)  # max temperature is 2 least is 0
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": sources}, 
        qa_template=QA_PROMPT, 
        question_generator_template=CONDENSE_PROMPT
    )  # 9 is the max sources
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=model, 
        retriever=retriever, 
        return_source_documents=True
    )
    return qa
