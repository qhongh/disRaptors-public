from langchain_core.documents import Document
from llama_index.core.schema import Document as LDocument
from langchain_core.language_models import BaseLanguageModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from llama_index.core.node_parser import SentenceSplitter
from langchain_core.messages import SystemMessage, HumanMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from functools import reduce
import asyncio
from datetime import datetime
import re
import pickle
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from typing import Any, List
from collections import defaultdict
from disraptors.rag.reddit_api import Reddit, format_search_results
from disraptors.rag.utils import (
    load_config,
    get_batches,
    llamaindex_textnode_to_document,
    monitor_llm_response_metadata,
    load_persisted_runs,
    persist_data,
)
from disraptors.rag.reddit_ingestor import SUBREDDITS


"""
TODO Trend Pagetest
- Get most mentioned topics in the last 2 years
- Get the most mentioned product and summarize sentiments
"""


class TrendPrompts:
    """Prompts for Trending Topics. Define blocks and spaces clearly. Resist the urge to use triple quotes in prompt."""

    system_message = SystemMessage(
        "You are a investment product specialist that does research on social media to learn about retail investors' preference."
    )
    entity_types = ["stock", "etf", "mutual fund", "index", "asset class", "industry sector", "currency"]

    @classmethod
    def extract_entities(cls) -> Any:
        assert len(cls.entity_types) > 1, "Please provide at least 2 entity types."
        user_message: HumanMessage = HumanMessagePromptTemplate.from_template(
            "You understand common asset classes are stock, bond, cash, foreign currency, real estate, infrastructure and commodity.\n"
            f"Below is a post from social media. Extract {' ,'.join(cls.entity_types[:-1])} and {cls.entity_types[-1]} entities in the python dict format.\n"
            "<post>\n"
            "Post: {post}\n"
            "</post>\n"
            "Please do not include any preamble or conclusions before or after the python dict. Please make sure keys in the python dict are not plural forms."
        )
        return ChatPromptTemplate.from_messages([cls.system_message, user_message])

    @classmethod
    def extract_topics_v2(cls) -> Any:
        user_message: HumanMessage = HumanMessagePromptTemplate.from_template(
            "Below lists a post from social media and important entity dict extracted from the post. Summarize investment topics by using hints from the extracted entity dict.\n"
            "Output the result in the python dict format, with investment topics as keys and summarizations as values.\n"
            "<post>\n"
            "Post: {post}\n"
            "</post>\n"
            "<entities>\n"
            "Extracted Entities: {entities}\n"
            "</entities>\n"
            "Please do not include any preamble or conclusions before or after the python dict."
        )
        return ChatPromptTemplate.from_messages([cls.system_message, user_message])

    @classmethod
    def extract_topics(cls) -> Any:
        user_message: HumanMessage = HumanMessagePromptTemplate.from_template(
            "The context is a post from social media. Summarize the investment key topics in the python dict format. The keys of the python dict are the key topics, and the values are summarizations.\n"
            "<post>\n"
            "Post: {post}\n"
            "</post>\n"
            "Please do not include any preamble or conclusions before or after the python dict."
        )
        return ChatPromptTemplate.from_messages([cls.system_message, user_message])

    @classmethod
    def filter_posts(cls) -> Any:
        user_message: HumanMessage = HumanMessagePromptTemplate.from_template(
            "<post>\n"
            "Post: {post}\n"
            "</post>\n"
            "Is this social media post relevant to investment products? Please only answer 'yes' or 'no'."
        )
        return ChatPromptTemplate.from_messages([cls.system_message, user_message])

    @classmethod
    def summarize_popular_opinions(cls) -> Any:
        user_message: HumanMessage = HumanMessagePromptTemplate.from_template(
            "You are given a list of summarizations from different social media posts. List the most popular 3 opinions in the python dict format. The keys of the python dict are the popular opinions, and the values are given summarizations from the context.\n"
            "<list>\n"
            "Post: {post}\n"
            "</list>\n"
            "Please do not include any preamble or conclusions before or after the python dict."
        )
        return ChatPromptTemplate.from_messages([cls.system_message, user_message])

    @classmethod
    def name_opinion_group(cls) -> Any:
        user_message: HumanMessage = HumanMessagePromptTemplate.from_template(
            "You are given a list of topics. Summarize them in one phrase.\n"
            "<list>\n"
            "Topics: {topics}\n"
            "</list>\n"
            "Please do not include any preamble or conclusions before or after the phrase."
        )
        return ChatPromptTemplate.from_messages([cls.system_message, user_message])


def eval_structured_output(output: str):
    cleaned_output = re.sub(r"(?<=[A-Za-z])'(?=[A-Za-z'])", "", output.strip())
    try:
        return eval(cleaned_output)
    except Exception as e:
        print(cleaned_output, output, e)
        return {}


class Trend:
    secrets: dict = load_config("secret.config")
    default_llm: BaseLanguageModel = ChatAnthropic(
        model="claude-3-sonnet-20240229", api_key=secrets["claude"]["api_key"], temperature=0
    )
    dense_encoder = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=secrets["openai"]["api_key"])
    # Keep long context window rather than split every sentence
    splitter: SentenceSplitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    # save cost by not re-embedding the same data in the same day
    persist_folder = Path(__file__).parent / "trend_pod"

    def __init__(
        self,
        limit: int = 50,
        llm: BaseLanguageModel = None,
        batch_size: str = 10,
        date: str = None,
        refresh: bool = False,
    ) -> None:
        self.llm = llm or self.default_llm
        self.batch_size = batch_size
        self.limit = limit
        self.refresh = refresh
        self.chunks = []
        self.date = datetime.now().strftime("%Y-%m-%d") if date is None else date
        self.persist_folder = self.persist_folder / self.date / f"{self.limit}"
        self.persist_folder.mkdir(exist_ok=True, parents=True)
        self.ent_keys = [self.clean_ent_key(k) for k in TrendPrompts.entity_types]
        self.logger = logging.getLogger(__name__)

    def get_raw_hot_posts(self, refresh: bool = False):
        ppath = self.persist_folder / "raw_hot_posts.pkl"
        res = load_persisted_runs(persist_path=ppath, refresh=self.refresh or refresh)
        if res is not None:
            return res

        reddit = Reddit()
        subreddits = "+".join(SUBREDDITS)

        posts = reddit.get_hot_posts(subreddit=subreddits, limit=self.limit)
        additional_metadata = {
            "tag": "hot",
            "ingest_date": self.date,
        }

        posts = format_search_results(
            posts,
            to="document",
            include_comments=True,
            only_toplevel_comments=True,
        )
        if additional_metadata is not None:
            for post in posts:
                post.metadata.update(additional_metadata)

        chunks = self.splitter([LDocument.from_langchain_format(post) for post in posts])
        chunks = [llamaindex_textnode_to_document(n) for n in chunks]

        persist_data(chunks, ppath)
        return chunks

    def batch_extract_entities(self, batch: List[Document]):
        extract_entity_chain = TrendPrompts.extract_entities() | self.llm
        for doc in batch:
            rsp = extract_entity_chain.invoke(doc.page_content)
            monitor_llm_response_metadata(rsp.response_metadata)
            try:
                ents = eval_structured_output(rsp.content)
            except Exception as e:
                self.logger.error(f"Error extracting entities {doc.page_content}: {rsp.content}: {e}")
                ents = {}

            cleaned_ents = {}
            for k, v in ents.items():
                if not isinstance(v, list):
                    self.logger.error(f"Unexpected entity format: {k}: {v}")
                    continue

                if len(v) != 0:
                    ck = self.clean_ent_key(k)
                    if ck in self.ent_keys:
                        cleaned_ents[self.clean_ent_key(k)] = v
            doc.metadata["entities"] = cleaned_ents
        return batch

    @staticmethod
    def clean_ent_key(k: str):
        return k.strip().replace(" ", "_")

    def extract_entities(self):
        async def abatch_extract_entities():
            batches = get_batches(self.get_raw_hot_posts(), batch_size=self.batch_size)
            res = await asyncio.gather(
                *[asyncio.to_thread(self.batch_extract_entities, batch=batch) for batch in batches]
            )
            return reduce(lambda a, b: a + b, res)

        return asyncio.run(abatch_extract_entities())

    def get_hot_posts(self, refresh: bool = False):
        ppath = self.persist_folder / "hot_posts.pkl"
        res = load_persisted_runs(persist_path=ppath, refresh=self.refresh or refresh)
        if res is not None:
            return res

        docs = self.extract_entities()
        filtered_docs = []
        for d in docs:
            if len(d.metadata["entities"]) != 0:
                filtered_docs.append(d)
        self.logger.info(
            f"Filtered {len(docs) - len(filtered_docs)} / {len(docs)} docs with no investment related entities"
        )

        persist_data(filtered_docs, ppath)
        return filtered_docs

    def batch_summarize_chunk_topic(self, batch: List[Document]):
        extract_topic_chain = TrendPrompts.extract_topics() | self.llm
        for doc in batch:
            output = extract_topic_chain.invoke(doc.page_content)
            monitor_llm_response_metadata(output.response_metadata)
            doc.metadata["topics"] = eval_structured_output(output.content)
        return batch

    def summarize_chunk_topic(self, refresh: bool = False):
        ppath = self.persist_folder / "chunk_topics.pkl"
        res = load_persisted_runs(persist_path=ppath, refresh=self.refresh or refresh)
        if res is not None:
            return res

        important_ent_types = {"stock", "etf", "mutual_fund", "index"}
        filtered_post = [
            p
            for p in self.get_hot_posts()
            if len(set(p.metadata["entities"].keys()).intersection(important_ent_types)) != 0
        ]

        async def asummmarize_chunk_topic():
            batches = get_batches(filtered_post, batch_size=self.batch_size)
            res = await asyncio.gather(
                *[asyncio.to_thread(self.batch_summarize_chunk_topic, batch=batch) for batch in batches]
            )
            return reduce(lambda a, b: a + b, res)

        chunk_topics = asyncio.run(asummmarize_chunk_topic())

        persist_data(chunk_topics, ppath)
        return chunk_topics

    def get_popular_opinions(self, refresh: bool = False):
        ppath = self.persist_folder / "popular_opinions.pkl"
        res = load_persisted_runs(persist_path=ppath, refresh=self.refresh or refresh)
        if res is not None:
            return res

        from sklearn.manifold import TSNE
        from sklearn.cluster import HDBSCAN
        from sklearn.metrics.pairwise import cosine_similarity

        chunk_topics = self.summarize_chunk_topic(refresh=False)
        zipped_topics = reduce(
            lambda a, b: a + b,
            [
                [
                    (f"{k} : {v}", v, ct.metadata["url"] + ct.metadata.get("comment_id", ""))
                    for k, v in ct.metadata["topics"].items()
                ]
                for ct in chunk_topics
            ],
        )
        topics_with_keys, topics, topic_urls = zip(*zipped_topics)

        # Cluster topics
        embeddings = self.dense_encoder.embed_documents(topics_with_keys)
        reps = np.array(embeddings) @ np.array(embeddings).T
        # TODO Implement hyperparameter selection logic
        tsne = TSNE(n_components=3, perplexity=10, random_state=0)
        reps = tsne.fit_transform(reps)
        hdb = HDBSCAN(min_cluster_size=5, algorithm="ball_tree")
        hdb.fit(reps)
        # Summarize popular opinions
        sp = TrendPrompts.summarize_popular_opinions() | self.llm
        no = TrendPrompts.name_opinion_group() | self.llm

        labels = pd.Series(hdb.labels_).value_counts()

        def link_opinion_cohort(label_: int):
            if label_ == -1:
                return None

            cohort = list(np.array(topics)[np.where(hdb.labels_ == label_)[0]])
            cohort_embedding = None
            cohort_url = list(np.array(topic_urls)[np.where(hdb.labels_ == label_)[0]])
            op = eval_structured_output(sp.invoke("\n".join(cohort)).content)
            if len(op) == 0:
                return None

            op_with_source = defaultdict(list)
            for k, v in op.items():
                if isinstance(v, str):
                    v = [v]
                for i in v:
                    # link to source
                    try:
                        op_with_source[k].append(cohort_url[cohort.index(i)])
                    except ValueError:
                        if cohort_embedding is None:
                            cohort_embedding = self.dense_encoder.embed_documents(cohort)
                        ie = self.dense_encoder.embed_documents([i])
                        sim = cosine_similarity(ie, cohort_embedding)
                        op_with_source[k].append(cohort_url[sim.argmax()])
                op_with_source[k] = list(set(op_with_source[k]))
            cohort_name = no.invoke("\n".join(op.keys())).content
            return cohort_name, op_with_source

        async def aget_popular_opinions():
            return await asyncio.gather(*[asyncio.to_thread(link_opinion_cohort, label_) for label_ in labels.index])

        opinions = dict()
        res = asyncio.run(aget_popular_opinions())
        for item in res:
            if item is None:
                continue
            cohort_name, op_with_source = item
            opinions[cohort_name] = op_with_source

        opinions = self.sort_popular_opinions(opinions)
        persist_data(opinions, ppath)
        return opinions

    def sort_popular_opinions(self, opinions: dict):
        from collections import OrderedDict

        sorted_opinions = OrderedDict(
            sorted(
                opinions.items(),
                key=lambda x: reduce(lambda a, b: a + b, [len(l) for l in x[1].values()]),
                reverse=True,
            )
        )
        final_sorted_opinions = OrderedDict()
        for k, v in sorted_opinions.items():
            final_sorted_opinions[k] = OrderedDict(sorted(v.items(), key=lambda x: len(x[1]), reverse=True))
        return final_sorted_opinions

    def format_popular_opinions(self):
        opinions = self.get_popular_opinions()
        formatted_opinions = []
        for i, (k, v) in enumerate(opinions.items()):
            formatted_opinions.append(f"{i + 1}. {k}:")
            for k1, v1 in v.items():
                markdown_urls = [f"[source {j + 1}]({url})" for j, url in enumerate(v1)]
                formatted_opinions.append(f"\t- {k1}. ({', '.join(markdown_urls)})")
        return "\n".join(formatted_opinions)
