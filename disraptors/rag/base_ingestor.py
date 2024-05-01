from pinecone import Pinecone, ServerlessSpec
from typing import Optional, List
from functools import cached_property
import logging
from langchain_core.documents import Document
from llama_index.core.schema import Document as LDocument
from disraptors.rag.pinecone_retriever import PineConeRetriever
from disraptors.rag.utils import load_config, llamaindex_textnode_to_document
from pinecone_text.sparse import BM25Encoder
import logging


class PineConeBaseIngestor:
    _dimension: Optional[int] = None
    default_chunk_size: Optional[int] = 800
    default_chunk_overlap: Optional[int] = 100
    default_window_size: Optional[int] = 3

    def __init__(
        self,
        index_name: Optional[str] = "quicktest",
        namespace: Optional[str] = None,
        splitter: Optional[str] = "window",
        **splitter_kwargs,
    ) -> None:
        self.index_name = index_name
        self.namespace = namespace
        self.splitter = splitter
        self.splitter_kwargs = splitter_kwargs
        self.secrets = load_config("secret.config")
        self.logger = logging.getLogger(__name__)

    @cached_property
    def pinecone(self):
        return Pinecone(api_key=self.secrets["pinecone"]["api_key"])

    @cached_property
    def available_index_names(self):
        return [i.name for i in self.pinecone.list_indexes()]

    @cached_property
    def dense_encoder(self):
        from langchain_openai import OpenAIEmbeddings

        model = "text-embedding-ada-002"
        self._dimension = 1536
        return OpenAIEmbeddings(openai_api_key=self.secrets["openai"]["api_key"], model=model)

    @cached_property
    def sparse_encoder(self):
        return BM25Encoder().default()

    def is_noisy_doc(self, doc: str) -> bool:
        return len(self.sparse_encoder.encode_documents(doc.page_content)["values"]) == 0

    def create_index(self, index_name: str):
        self.pinecone.create_index(
            name=index_name,
            dimension=self._dimension,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    @cached_property
    def index(self):
        if self.index_name not in self.available_index_names:
            self.logger.info(f"\nIndex {self.index_name} not found. Creating new index.")
            self.create_index(self.index_name)

        return self.pinecone.Index(self.index_name)

    @cached_property
    def vector_store(self):
        return PineConeRetriever(
            embeddings=self.dense_encoder,
            sparse_encoder=self.sparse_encoder,
            index=self.index,
            namespace=self.namespace,
        )

    @cached_property
    def recursive_text_splitter(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from disraptors.rag.utils import DEFAULT_CHUNK_SEPARATOR

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.splitter_kwargs.get("chunk_size", self.default_chunk_size),
            chunk_overlap=self.splitter_kwargs.get("chunk_overlap", self.default_chunk_overlap),
            length_function=len,
            separators=DEFAULT_CHUNK_SEPARATOR,
            is_separator_regex=False,
        )
        return text_splitter.split_documents

    @cached_property
    def window_text_splitter(self):
        from llama_index.core.node_parser import SentenceWindowNodeParser

        # create the sentence window node parser w/ default settings
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=self.splitter_kwargs.get("window_size", self.default_window_size),
        )

        def _splitter(docs: List[Document]):
            ldocs = [LDocument.from_langchain_format(d) for d in docs]
            nodes = node_parser.get_nodes_from_documents(ldocs)
            return [llamaindex_textnode_to_document(n) for n in nodes]

        return _splitter

    @cached_property
    def sentence_splitter(self):
        from llama_index.core.node_parser import SentenceSplitter

        lsplitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

        def _splitter(docs: List[Document]):
            chunks = lsplitter([LDocument.from_langchain_format(doc) for doc in docs])
            chunks = [llamaindex_textnode_to_document(chunk) for chunk in chunks]
            return chunks

        return _splitter

    @cached_property
    def text_splitter(self):
        assert self.splitter in [
            "window",
            "recursive",
            "sentence",
        ], "Splitter must be one of 'window', 'sentence' or 'recursive'"
        if self.splitter == "window":
            return self.window_text_splitter
        elif self.splitter == "recursive":
            return self.recursive_text_splitter
        else:
            return self.sentence_splitter
