from typing import Optional, List
from langchain_core.documents import Document
import logging
import asyncio
import re
from disraptors.rag.base_ingestor import PineConeBaseIngestor
from disraptors.rag.pinecone_retriever import NAMESPACE
from disraptors.rag.regex_patterns import TICKER, VANGUARD_PRODUCTS
from disraptors.rag.utils import get_batches
from pathlib import Path
import json

# TODO
# - Add Twitter API pull instead of reading from local files

def format_twitter_search_results(tweets: List[dict]):
    # TODO move to the twitter api class
    tweets = tweets.copy()
    cleaned_tweets = []
    for tw in tweets:
        to_remove = tw["entities"].pop("urls", None)
        if to_remove is None:
            ctext = tw["text"]
        else:
            places = sorted([(x["start"], x["end"]) for x in to_remove], key=lambda x: x[0])
            ctext = ""
            last_end = 0
            for start, end in places:
                ctext += tw["text"][last_end:start]
                last_end = end
            ctext += tw["text"][end:]

        ctext = re.sub(r"(@[a-zA-Z0-9_]+\s*)+", "", ctext)

        metadata = {
            "post_id": tw["author_id"],
            "post_created_utc": tw["created_at"],
            "author_id": tw["author_id"],
        }
        metadata.update({k: str(v) for k, v in tw["entities"].items()})
        metadata.update(tw["public_metrics"])
        cleaned_tweets.append(Document(page_content=ctext, metadata=metadata))
    return cleaned_tweets


class TwitterIngestor(PineConeBaseIngestor):
    """Ingestor for Twitter data."""

    namespace: Optional[str] = NAMESPACE
    include_comments: Optional[bool] = True
    default_time_filter: Optional[str] = "all"
    default_limit: Optional[int] = 1000

    def __init__(
        self,
        index_name: Optional[str] = "quicktest",
        namespace: Optional[str] = None,
        splitter: Optional[str] = "sentence",
        **splitter_kwargs,
    ):

        super().__init__(
            index_name=index_name,
            namespace=namespace or self.namespace,
            splitter=splitter,
            **splitter_kwargs,
        )
        self.logger = logging.getLogger(__name__)

    def add_metadata(self, doc: Document):
        doc.metadata["tickers"] = list(set(re.findall(TICKER, doc.page_content)))
        doc.metadata["vanguard_product"] = list(set(re.findall(VANGUARD_PRODUCTS, doc.page_content)))
        doc.metadata["source"] = "twitter"
        return doc

    def split_posts(self, tweets: List[dict]):
        tweets = format_twitter_search_results(tweets)
        chunks = self.text_splitter(tweets)

        cleaned_chunks = []
        for ck in chunks:
            if not self.is_noisy_doc(ck):
                # remove documents with only filler words
                ck = self.add_metadata(ck)
                cleaned_chunks.append(ck)
        return cleaned_chunks

    def load_twitter_data(self, folder: str) -> List:
        paths = Path(folder).rglob("*.json")
        data = []
        for path in paths:
            with open(path, "r") as f:
                data.extend(json.load(f)["data"])
        return data

    def ingest_batch(self, batch: List):
        chunks = self.split_posts(batch)
        self.vector_store.add_documents(chunks, namespace=self.namespace)
        return chunks

    def ingest_from_folder(self, folder: str):
        batch_size = 100

        data = self.load_twitter_data(folder)

        async def aingest():
            res = await asyncio.gather(
                *[
                    asyncio.to_thread(self.ingest_batch, batch=batch)
                    for batch in get_batches(data, batch_size=batch_size)
                ]
            )
            return res

        return asyncio.run(aingest())


if __name__ == "__main__":
    ti = TwitterIngestor(index="quicktest", namespace="temp")
    tweets = ti.ingest_from_folder("disraptors/rag/tweets/")
