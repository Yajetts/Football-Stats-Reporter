from typing import List
from pydantic import BaseModel, Field
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
import os
from dotenv import load_dotenv
import datetime

# Load environment variables from .env file
load_dotenv()

# Basic document structure
class Document(BaseModel):
    content: str = Field(..., description="The content of the document")
    metadata: dict = Field(
        default_factory=dict, description="Document metadata"
    )

# Structure for query responses
class QueryResult(BaseModel):
    answer: str = Field(..., description="Response to the query")
    source_nodes: List[str] = Field(
        ..., description="Source references used"
    )

class FootballStatsReporter:
    def __init__(
        self,
        data_path: str,
        index_path: str = "index",
    ):
        """
        Initialize the AI Assistant
        :param data_path: Path to your document directory
        :param index_path: Path where the vector index will be stored
        """
        self.data_path = data_path
        self.index_path = index_path
        
        # Customize this prompt for your use case
        self.system_prompt = """
         You are a helpful AI Football Stats Reporter with access to a database of football information:
        1. Your role is to find a particular statistic related to a football player or team and to display it as a repsonse.
        2. You are responsible to find reliable and trustable data from your tool to access the database and provide the requested information.
        3. You are only supposed to answer queries related to football statistics and no other topic or sport.
        4. Responses will only be diaplayed for queries related to football
        """

        self.configure_settings()
        self.index = None
        self.agent = None
        self.load_or_create_index()

    def configure_settings(self):
        """Configure LLM and embedding settings"""
        # Replace with your preferred LLM
        Settings.llm = Groq(model="llama-3.1-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))  # Add your LLM configuration here
        # Replace with your preferred embedding model
        Settings.embed_model = JinaEmbedding(
            api_key=os.getenv("JINA_API_KEY"),
            model="jina-embeddings-v2-base-en",
        )  # Add your embedding model configuration here

    def load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.create_index()
        self._create_agent()

    def load_index(self):
        """Load existing vector index"""
        storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
        self.index = load_index_from_storage(storage_context)

    def create_index(self):
        """Create new vector index from documents"""
        documents = SimpleDirectoryReader(
            self.data_path,
            recursive=True,
        ).load_data()
        if not documents:
            raise ValueError("No documents found in specified path")
        self.index = VectorStoreIndex.from_documents(documents)
        self.save_index()

    def _create_agent(self):
        """Set up the agent with custom tools"""
        query_engine = self.index.as_query_engine(similarity_top_k=10)
        
        # Basic search tool
        search_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="document_search",
                description="Get information about football statistics, matches and players from the knowledge base",
            ),
        )
        # Example of a custom tool
        def custom_function(query: str) -> str:
            """Custom functionality to return the current date and time"""
            if "current date" in query.lower():
                return f"The current date and time is: {datetime.datetime.now()}"
            # return f"Custom response for: {query}"
        
        custom_tool = FunctionTool.from_defaults(
            fn=custom_function,
            name="custom_tool",
            description="Returns the current date and time if asked"
        )

        # Initialize the agent with tools
        self.agent = ReActAgent.from_tools(
            [search_tool],
            verbose=True,
            system_prompt=self.system_prompt,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
        )

    def query(self, query: str) -> QueryResult:
        """
        Process a query and return results
        :param query: User's question or request
        :return: QueryResult with answer and sources
        """
        if not self.agent:
            raise ValueError("Agent not initialized")
        response = self.agent.chat(query)
        return QueryResult(
            answer=response.response,
            source_nodes=[],
        )

    def save_index(self):
        """Save the vector index to disk"""
        os.makedirs(self.index_path, exist_ok=True)
        self.index.storage_context.persist(persist_dir=self.index_path)

# # Example usage
# if __name__ == "__main__":
#     assistant = FootballStatsReporter(
#         data_path="./your_data_directory",
#         index_path="your_index_directory"
#     )
#     result = assistant.query("Your question here")
#     print(result.answer)