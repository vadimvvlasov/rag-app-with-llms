"""
Public API for the rag-intro package.

Re-exports all public classes from sub-modules.  Imports are guarded so that
modules which have not been implemented yet do not prevent the rest of the
package from loading.
"""

from .ingest import ElasticsearchIndex, FaqHttpLoader, MinsearchIndex, SqliteIndex
from .interfaces import DataLoader, LLMClient, SearchIndex

try:
    from .llm import OllamaClient, OpenAIClient, OpenRouterClient
except ModuleNotFoundError:
    pass

try:
    from .rag import RAGBase
except ModuleNotFoundError:
    pass
