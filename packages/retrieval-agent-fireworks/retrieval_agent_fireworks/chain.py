from typing import List

from langchain import hub
from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.tools.render import render_text_description
from langchain.tools.retriever import create_retriever_tool
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain_community.chat_models.fireworks import ChatFireworks
from langchain_fireworks import ChatFireworks
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma
from langchain_nomic import NomicEmbeddings

MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"

# class ArxivRetriever(BaseRetriever, ArxivAPIWrapper):
#     """`Arxiv` retriever.
#
#     It wraps load() to get_relevant_documents().
#     It uses all ArxivAPIWrapper arguments without any change.
#     """
#
#     get_full_documents: bool = False
#
#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
#     ) -> List[Document]:
#         try:
#             if self.is_arxiv_identifier(query):
#                 results = self.arxiv_search(
#                     id_list=query.split(),
#                     max_results=self.top_k_results,
#                 ).results()
#             else:
#                 results = self.arxiv_search(  # type: ignore
#                     query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results
#                 ).results()
#         except self.arxiv_exceptions as ex:
#             return [Document(page_content=f"Arxiv exception: {ex}")]
#         docs = [
#             Document(
#                 page_content=result.summary,
#                 metadata={
#                     "Published": result.updated.date(),
#                     "Title": result.title,
#                     "Authors": ", ".join(a.name for a in result.authors),
#                 },
#             )
#             for result in results
#         ]
#         return docs
#
#
# # Set up tool(s)
# description = (
#     "A wrapper around Arxiv.org "
#     "Useful for when you need to answer questions about Physics, Mathematics, "
#     "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
#     "Electrical Engineering, and Economics "
#     "from scientific articles on arxiv.org. "
#     "Input should be a search query."
# )
# arxiv_tool = create_retriever_tool(ArxivRetriever(), "arxiv", description)


''' LCEL documentation tool '''
# embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                   model_kwargs={'device': 'cuda'})
vectorstore = Chroma(embedding_function=embeddings, persist_directory='./chroma_db')
vectorstore_retriever = vectorstore.as_retriever()
vectorstore_tool_description = (
    # 'A tool that looks up documentation about the LangChain Expression Language.'
    'A tool that looks up into HuggingFace documentation about the parameters that would help with the pictures configuration.'
)
vectorstore_tool = create_retriever_tool(vectorstore_retriever, name="docstore",
                                         description=vectorstore_tool_description)


@tool
def increase_brightness(brightness: int, increment: int = 5) -> int:
    """
        Useful when you want to increase the value of variable brightness.
        :param brightness: Current value of the variable which is 0.
        :param increment: Increment value (default=5).
        :return: Updated value after increment.
        """
    print(type(brightness), type(increment))
    return int(brightness) + increment


@tool
def decrease_brightness(brightness: int, increment: int = -5) -> int:
    """
        Useful when you want to decrease the value of variable brightness.
        :param brightness: Current value of the variable which is 0.
        :param increment: Increment value (default=-5).
        :return: Updated value after increment.
        """
    print(type(brightness), type(increment))
    return int(brightness) + increment


# tools = [vectorstore_tool, increase_brightness, decrease_brightness]
tools = [increase_brightness, decrease_brightness]

# Set up LLM
llm = ChatFireworks(
    model_name=MODEL_ID,
    max_tokens=2048,
    temperature=0,
    cache=True,
)

# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# define the agent
model_with_stop = llm.bind(stop=["\nObservation"])
init_prompt = ("You are an assistant that is helping with picture editing. "
               "You are able to increase or decrease the brightness of a photo."
               " You can also use it to look up into HuggingFace documentation about the parameters that would "
               "help with the pictures configuration. To use the agent, you can input 'increase brightness', "
               "'decrease brightness', or a query for the documentation."
               "If you don't know the answer, just say that you don't know. \
                Use three sentences maximum and keep the answer concise")
agent = (
        {
            "input": lambda x: x["input"] if x['input'] else init_prompt,
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | model_with_stop
        | ReActJsonSingleInputOutputParser()
)


class InputType(BaseModel):
    input: str


# instantiate AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
).with_types(input_type=InputType)
