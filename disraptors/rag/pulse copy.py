from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from functools import reduce
import asyncio
from datetime import datetime
import re
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, List, Optional
from disraptors.rag.regex_patterns import VANGUARD_PRODUCTS
from disraptors.rag.pinecone_retriever import PineConeRetriever
from disraptors.rag.utils import (
    load_config,
    get_batches,
    monitor_llm_response_metadata,
    lookback_epoch_time,
    load_persisted_runs,
    persist_data,
)

"""
TODO Trend Pagetest
- Get most mentioned topics in the last 2 years
- Get the most mentioned product and summarize sentiments
"""


class PulsePrompts:
    """Prompts for Trending Topics. Define blocks and spaces clearly. Resist the urge to use triple quotes in prompt."""

    system_message = SystemMessage(
        "You are a investment product specialist that does research on social media to learn about retail investors' preference."
    )
    divider: str = "\n" + "=" * 10 + "\n"

    @classmethod
    def qa_prompt(cls) -> Any:
        user_message: HumanMessage = HumanMessagePromptTemplate.from_template(
            "You are given a context and a question. Answer the question based on the context. If you are not able to find the answer in the context, just say I don't know.\n"
            "The context is a list of posts and comments pulled from social media based on the question separated by equal signs.\n"
            "<context>\n"
            "Context: {context}\n"
            "</context>\n"
            "<question>\n"
            "Question: {question}\n"
            "</question>\n"
        )
        return ChatPromptTemplate.from_messages([cls.system_message, user_message])

    @classmethod
    def extract_prompt(cls) -> Any:
        user_message: HumanMessage = HumanMessagePromptTemplate.from_template(
            "Given the following context and question, extract any part of the context *AS IS* that is relevant to answer the question. If none of the context is relevant return NONE.\n"
            "Remember, *DO NOT* edit the extracted parts of the context.\n"
            "<context>\n"
            "Context: {context}\n"
            "</context>\n"
            "<question>\n"
            "Question: {question}\n"
            "</question>\n"
            "Skip the preamble and provide only the relevant part of the context.\n"
        )
        return ChatPromptTemplate.from_messages([cls.system_message, user_message])


def eval_structured_output(output: str):
    cleaned_output = re.sub(r"(?<=[A-Za-z])'(?=[A-Za-z'])", "", output.strip())
    try:
        return eval(cleaned_output)
    except Exception as e:
        print(cleaned_output, output, e)
        return {}


class Pulse:
    secrets: dict = load_config("secret.config")
    default_llm: BaseLanguageModel = ChatAnthropic(
        model="claude-3-sonnet-20240229", api_key=secrets["claude"]["api_key"], temperature=0
    )
    # save cost by not re-embedding the same data in the same day
    persist_folder = Path(__file__).parent / "pulse_pod"

    def __init__(
        self,
        retriever: PineConeRetriever,
        llm: BaseLanguageModel = None,
        date: str = None,
        refresh: bool = False,
    ) -> None:
        self.llm = llm or self.default_llm
        self.retriever = retriever
        self.date = date or datetime.today().strftime("%Y-%m-%d")
        self.persist_folder = self.persist_folder / self.date
        self.refresh = refresh
        self.persist_folder.mkdir(parents=True, exist_ok=True)

    def compress_batch(self, batch: List[Document], query: str):
        compress_chain = PulsePrompts.extract_prompt() | self.llm
        compressed = []
        for chunk in batch:
            ans = compress_chain.invoke({"context": chunk.metadata["window"], "question": query}).content
            if ans != "NONE":
                compressed.append(Document(page_content=ans, metadata=chunk.metadata))
        return compressed

    def rag_qa(
        self, query: str, retrieve_query: Optional[str] = None, time_delta: dict = {"years": 2}, **kwargs
    ) -> List:
        filter_ = kwargs.get("filter_", {})
        filter_["post_created_utc"] = {"$gte": int(lookback_epoch_time(**time_delta))}

        if retrieve_query is None:
            retrieve_query = query

        try:
            chunks = self.retriever.get_relevant_documents(retrieve_query, filter_=filter_, **kwargs)
        except Exception as e:
            self.logger.error(f"Error retrieving query {retrieve_query}: {e}")
            return "No data found."

        if len(chunks) == 0:
            return "No data found."

        batch_size = 10

        async def compress_chunks():
            return await asyncio.gather(
                *[
                    asyncio.to_thread(self.compress_batch, batch=batch, query=retrieve_query)
                    for batch in get_batches(chunks, batch_size=batch_size)
                ]
            )

        nested_list = asyncio.run(compress_chunks())
        compressed = reduce(lambda x, y: x + y, nested_list)

        if len(compressed) == 0:
            return "No data found."

        qa_chain = PulsePrompts.qa_prompt() | self.llm
        res = qa_chain.invoke(
            {"context": PulsePrompts.divider.join([d.page_content for d in compressed]), "question": query}
        )
        monitor_llm_response_metadata(res.response_metadata)
        return res.content

    def get_product_synopsis(self, products: List[str], time_delta: dict = {"years": 2}, refresh: bool = False):
        top_k = 100
        alpha = 0.5

        time_tag = f"{self.date}_lookback_" + "_".join([f"{k}_{v}" for k, v in time_delta.items()])
        ppath = self.persist_folder / "products"
        ppath.mkdir(parents=True, exist_ok=True)

        product_synopsis = {}
        for product in products:
            path_ = ppath / f"{product}_{time_tag}.pkl"
            res = load_persisted_runs(persist_path=path_, refresh=refresh or self.refresh)
            if res is not None:
                product_synopsis[product] = {"date": self.date, "synopsis": res, "tag": time_tag}
                continue

            sp = self.rag_qa(
                query=f"Summarize opinions on {product}.",
                retrieve_query=product,
                time_delta=time_delta,
                top_k=top_k,
                alpha=alpha,
            )
            product_synopsis[product] = {"date": self.date, "synopsis": sp, "tag": time_tag}
            persist_data(data=product_synopsis[product], persist_path=path_)
        return product_synopsis

    def get_vg_product_synopsis(
        self, products: Optional[List[str]] = None, time_delta: dict = {"years": 2}, refresh: bool = False
    ):
        time_tag = f"{self.date}_lookback_" + "_".join([f"{k}_{v}" for k, v in time_delta.items()])
        ppath = self.persist_folder / f"vg_products_{time_tag}.pkl"

        res = load_persisted_runs(persist_path=ppath, refresh=refresh or self.refresh)
        if res is not None:
            return res

        if products is None:
            products = VANGUARD_PRODUCTS.strip(r"\b").split("|")
        products = [f"Vanguard ETF {product}" if len(product) < 3 else product for product in products]
        product_synopsis = self.get_product_synopsis(products=products, time_delta=time_delta)
        persist_data(data=product_synopsis, persist_path=ppath)
        return product_synopsis


if __name__ == "__main__":
    from pathlib import Path
    from datetime import datetime
    from disraptors.rag.utils import setup_logger, load_config
    from disraptors.rag.reddit_ingestor import RedditIngestor
    from langchain_anthropic import ChatAnthropic

    logger = setup_logger("run_pulse")

    secrets = load_config("secret.config")
    index_name = "disraptors-w"

    ri = RedditIngestor(index_name=index_name, splitter="window")
    retriever = ri.vector_store

    sonnet = ChatAnthropic(model="claude-3-sonnet-20240229", api_key=secrets["claude"]["api_key"], temperature=0)
    pulse = Pulse(retriever=retriever, llm=sonnet)
    # pulse.get_vg_product_synopsis(time_delta = {"years": 2}, refresh=False)
    sp = pulse.get_product_synopsis(products=["MEQT", "XCNS", "DCU", "FCNS", "FGRO"], refresh=False)
    persist_data(
        persist_path=Path(__file__).parent / "pulse_pod/2024-04-30/top_5_growth_2024-04-30_lookback_years_2.pkl",
        data=sp,
    )
