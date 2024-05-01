from langchain_community.retrievers.pinecone_hybrid_search import (
    PineconeHybridSearchRetriever,
)
from typing import List, Optional, Dict, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor


# Use one namespace for reddit and twitter, so that query can be done cross-platform
NAMESPACE = "reddit"

class PineConeRetriever(PineconeHybridSearchRetriever):
    """Wrapper for PineconeHybridSearchRetriever that allows for additional query parameters like filter to be passed"""

    embeddings: Embeddings
    sparse_encoder: Any
    index: Any
    top_k: int = 20
    alpha: float = 0.5  # alpha = 0 sparse only, = 1 dense only
    filter_: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None
    _expects_other_args: bool = True

    def add_documents(self, documents: list, namespace: Optional[str] = None):
        """Pinecone assigns ids according to the hash of the text, upsert with existing id will overwrite automatically, so duplicates of text will be handled."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts=texts, metadatas=metadatas, namespace=namespace)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs
    ) -> List[Document]:
        from pinecone_text.hybrid import hybrid_convex_scale

        sparse_vec = self.sparse_encoder.encode_queries(query)
        # convert the question into a dense vector
        dense_vec = self.embeddings.embed_query(query)
        # scale alpha with hybrid_scale
        alpha = kwargs.pop("alpha", self.alpha)
        dense_vec, sparse_vec = hybrid_convex_scale(dense_vec, sparse_vec, alpha)
        sparse_vec["values"] = [float(s1) for s1 in sparse_vec["values"]]
        # query pinecone with the query parameters
        top_k = kwargs.pop("top_k", self.top_k)
        include_metadata = kwargs.pop("include_metadata", True)
        namespace = kwargs.pop("namespace", self.namespace)

        filter_ = kwargs.pop("filter_", self.filter_)
        result = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=top_k,
            include_metadata=include_metadata,
            namespace=namespace,
            filter=filter_,
            **kwargs
        )
        final_result = []
        for res in result["matches"]:
            context = res["metadata"].pop("context")
            final_result.append(Document(page_content=context, metadata=res["metadata"]))
        # return search results as json
        return final_result
    

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        return await run_in_executor(
            None,
            self._get_relevant_documents,
            query,
            run_manager=run_manager.get_sync(),
        )
