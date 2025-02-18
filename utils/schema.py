from typing import List, Optional
from pydantic import BaseModel, Field
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.openai import OpenAI
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
        1. Your role is to find a particular statistic related to a football player or team and to display it as a response.
        2. You are responsible to find reliable and trustable data from your tool to access the database and provide the requested information.
        3. You are only supposed toanswer queries related to football statistics and no other topic or sport.
        4. Responses will only be displayed for queries related to football.
        """

        self.configure_settings()
        self.index = None
        self.agent = None
        self.load_or_create_index()

    def configure_settings(self):
        """Configure LLM and embedding settings"""
        # Use OpenAI GPT-3.5 Turbo
        Settings.llm = OpenAI(
            model="gpt-4",  # Use GPT-3.5 Turbo, or another model supported by OpenRouter
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
            }
        )
        # Use Jina Embeddings (or another embedding model)
        Settings.embed_model = JinaEmbedding(
            api_key=os.getenv("JINA_API_KEY"),
            model="jina-embeddings-v2-base-en",
        )

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

        try:
            # Attempt to generate a response using the agent
            response = self.agent.chat(query)
            return QueryResult(
                answer=response.response,
                source_nodes=[],
            )
        except Exception as e:
            # Handle specific errors
            error_message = f"Error generating response: {str(e)}"
            if "model_not_found" in str(e):
                error_message = "The requested model is not available. Please try again with a different model."
            elif "authentication_error" in str(e):
                error_message = "Authentication failed. Please check your API key."
            elif "rate_limit_exceeded" in str(e):
                error_message = "Rate limit exceeded. Please try again later."
            else:
                error_message = f"An unexpected error occurred: {str(e)}"

            # Return an error response
            return QueryResult(
                answer=error_message,
                source_nodes=[],
            )

    def save_index(self):
        """Save the vector index to disk"""
        os.makedirs(self.index_path, exist_ok=True)
        self.index.storage_context.persist(persist_dir=self.index_path)
