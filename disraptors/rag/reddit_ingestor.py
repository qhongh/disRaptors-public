from typing import Optional, List
from langchain_core.documents import Document
import logging
import time
import asyncio
import re
from disraptors.rag.base_ingestor import PineConeBaseIngestor
from disraptors.rag.pinecone_retriever import NAMESPACE
from disraptors.rag.reddit_api import Reddit, format_search_results
from disraptors.rag.regex_patterns import TICKER, VANGUARD_PRODUCTS
from disraptors.rag.utils import get_product_match_case_sensitivity
import datetime

SUBREDDITS = ["CanadianInvestor", "PersonalFinanceCanada"]

# TODO
# - Add Reddit tag to metadata


class RedditIngestor(PineConeBaseIngestor):
    namespace: Optional[str] = NAMESPACE
    include_comments: Optional[bool] = True
    default_time_filter: Optional[str] = "all"
    default_limit: Optional[int] = 1000

    def __init__(
        self,
        index_name: Optional[str] = "quicktest",
        namespace: Optional[str] = None,
        splitter: Optional[str] = "window",
        **splitter_kwargs,
    ):

        super().__init__(
            index_name=index_name,
            namespace=self.namespace or namespace,
            splitter=splitter,
            **splitter_kwargs,
        )
        self.reddit = Reddit()
        self.subreddits = "+".join(SUBREDDITS)
        self.reddit = Reddit()
        self.logger = logging.getLogger(__name__)

    def add_metadata(self, doc: Document):
        doc.metadata["tickers"] = list(set(re.findall(TICKER, doc.page_content)))
        doc.metadata["vanguard_product"] = list(set(re.findall(VANGUARD_PRODUCTS, doc.page_content)))
        doc.metadata["source"] = "reddit"
        return doc

    def split_posts(
        self,
        posts: list,
        only_toplevel_comments: bool = True,
        additional_metadata: Optional[dict] = None,
    ) -> List[Document]:
        self.logger.info(f"Start chunking {len(posts)} posts to documents...")
        tik = time.time()
        posts = format_search_results(
            posts,
            to="document",
            include_comments=self.include_comments,
            only_toplevel_comments=only_toplevel_comments,
        )
        if additional_metadata is not None:
            for post in posts:
                post.metadata.update(additional_metadata)

        docs = self.text_splitter(posts)
        self.logger.info(f"Finished formatting ({int(time.time()-tik)}s)")

        cleaned_docs = []
        for doc in docs:
            if not self.is_noisy_doc(doc):
                # remove documents with only filler words
                doc = self.add_metadata(doc)
                cleaned_docs.append(doc)
        return cleaned_docs

    def ingest_reddit_query(
        self,
        query: str,
        sort: str = "relevance",
        time_filter: Optional[str] = None,
        limit: Optional[int] = None,
        only_toplevel_comments: bool = True,
        phrase_check: Optional[str] = None,
        case_sensitivity: str = "strict",
        tag: Optional[str] = None,
    ):
        tik = time.time()
        posts = self.reddit.search_subreddit(
            subreddit=self.subreddits,
            query=query,
            sort=sort,
            time_filter=time_filter or self.default_time_filter,
            limit=limit or self.default_limit,
            only_toplevel_comments=only_toplevel_comments,
            phrase_check=phrase_check,
            case_sensitivity=case_sensitivity,
        )

        if len(posts) > 0:
            if tag is not None:
                for post in posts:
                    post["metadata"]["tag"] = tag

            docs = self.split_posts(posts, only_toplevel_comments=only_toplevel_comments)
            self.vector_store.add_documents(docs, namespace=self.namespace)
        self.logger.info(f"\nIngested Posts for Query: {query}({len(posts)} posts / {round(time.time() - tik)}s)\n")

    async def aingest_multiple_reddit_queries(
        self, queries: List[str], phrase_checks: List[Optional[str]] = None, **kwargs
    ):
        if not isinstance(phrase_checks, list):
            phrase_checks = [phrase_checks] * len(queries)

        await asyncio.gather(
            *[
                asyncio.to_thread(self.ingest_reddit_query, query=query, phrase_check=check, **kwargs)
                for query, check in zip(queries, phrase_checks)
            ]
        )

    def ingest_product_mentions(self, products: List[str] = None):
        reddit_kwargs = {
            "sort": "relevance",
            "time_filter": "all",
            "limit": 1000,
            "only_toplevel_comments": False,
        }

        tik = time.time()
        case_sensitivity_dict = get_product_match_case_sensitivity(products=products)
        for case_sensitivity, products in case_sensitivity_dict.items():
            if case_sensitivity == "strict":
                # If the case sensitivity of a product is strict, we will search for the product name and the product name with "Vanguard" and "ETF" appended to it.
                phrase_checks = []
                queries = []
                for product in products:
                    queries.append(f"Vanguard {product}")
                    queries.append(f"ETF {product}")
                    phrase_checks += [product, product]
            else:
                queries = products
                phrase_checks = products

            asyncio.run(
                self.aingest_multiple_reddit_queries(
                    queries=queries,
                    phrase_checks=phrase_checks,
                    case_sensitivity=case_sensitivity,
                    **reddit_kwargs,
                )
            )

        self.logger.info(
            f"\n-------------------- Finished Ingesting Product Mentions ({round(time.time() - tik)}s) --------------------------\n"
        )

    def ingest_phrases(self, phrases, phrase_checks, case_sensitivity, tag: Optional[str] = None):
        reddit_kwargs = {
            "sort": "relevance",
            "time_filter": "year",
            "limit": 1000,
            "only_toplevel_comments": False,
        }
        tik = time.time()
        asyncio.run(
            self.aingest_multiple_reddit_queries(
                queries=phrases,
                phrase_checks=phrase_checks,
                case_sensitivity=case_sensitivity,
                tag=tag,
                **reddit_kwargs,
            )
        )

        self.logger.info(
            f"\n-------------------- Finished Ingesting Phrases {phrases} ({round(time.time() - tik)}s) --------------------------\n"
        )

    def ingest_hot(self, only_toplevel_comments: bool = True, limit: Optional[int] = 50):
        tik = time.time()
        posts = self.reddit.get_hot_posts(subreddit=self.subreddits, limit=limit)

        docs = self.split_posts(
            posts,
            only_toplevel_comments=only_toplevel_comments,
            additional_metadata={
                "tag": "hot",
                "ingest_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            },
        )
        self.vector_store.add_documents(docs, namespace=self.namespace)
        self.logger.info(f"\nIngested hottest({len(posts)} posts / {round(time.time() - tik)}s)\n")
