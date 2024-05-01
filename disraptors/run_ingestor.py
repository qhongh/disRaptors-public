import logging
import sys
from pathlib import Path
from datetime import datetime
from disraptors.rag.pinecone_retriever import NAMESPACE


def setup_logger():
    Path("logs").mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/run_ingestor_{datetime.now().strftime('%Y-%m-%d %H-%M')}.log"),
        ],
    )

    # disable praw warning ouput
    for logger_name in ("praw", "prawcore"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    return logging.getLogger(__name__)


def ingest_from_reddit():
    from disraptors.rag.reddit_ingestor import RedditIngestor

    ri = RedditIngestor(index_name="disraptors-w", splitter="window")
    ri.ingest_product_mentions(products=["VLB"])
    # ri.ingest_phrases(phrases=["MER", "expense ratio", "management fee"], phrase_checks=["MER", None, None], case_sensitivity="ignore")
    # ri.ingest_hot(limit=50, only_toplevel_comments=False)

def ingest_from_twitter():
    from disraptors.rag.twitter_ingestor import TwitterIngestor
    ti = TwitterIngestor(index_name="disraptors-w", namespace=NAMESPACE, splitter="sentence")
    ti.ingest_from_folder("disraptors/rag/tweets/")



def main():
    setup_logger()
    # ingest_from_reddit()
    ingest_from_twitter()


if __name__ == "__main__":
    main()
