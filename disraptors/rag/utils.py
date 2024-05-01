import configparser
from pathlib import Path
from typing import List, Optional, Any
from disraptors.rag.regex_patterns import VANGUARD_PRODUCTS
import datetime
from dateutil.relativedelta import relativedelta
from langchain_core.documents import Document
from llama_index.core.schema import TextNode
import logging
import pickle
import sys


DEFAULT_DOCUMENT_SEPARATOR = "\n\n\n"
DEFAULT_PARAGRAPH_SEPARATOR = "\n\n"
DEFAULT_CHUNK_SEPARATOR = ["\n\n", "\n", " ", ""]
if DEFAULT_DOCUMENT_SEPARATOR not in DEFAULT_CHUNK_SEPARATOR:
    DEFAULT_CHUNK_SEPARATOR = [DEFAULT_DOCUMENT_SEPARATOR] + DEFAULT_CHUNK_SEPARATOR
DEFAULT_NO_OUPUT = "NO_OUTPUT"

REGEX_CASE_SENSITIVE_LEVELS = ["strict", "title", "ignore"]


def setup_logger(name: str = __name__):
    Path("logs").mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/{name}_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M')}.log"),
        ],
    )
    # disable praw warning ouput
    for logger_name in ("praw", "prawcore"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    return logging.getLogger(name)


def load_config(path: str):
    """Read setup config file"""

    file_path = Path(__file__).parent / path
    if Path(str(Path(file_path).resolve())).exists():
        config_obj = configparser.RawConfigParser()
        config_obj.read(file_path)
        return config_obj
    raise FileNotFoundError(f"Config file {file_path} not found.")


def get_product_match_case_sensitivity(products: Optional[List[str]] = None):
    if products is None:
        products = VANGUARD_PRODUCTS.strip(r"\b").split("|")
    strict_phrase_check = [p for p in products if len(p) < 3]
    loose_phrase_check = [p for p in products if len(p) >= 3]
    return {"strict": strict_phrase_check, "ignore": loose_phrase_check}


def lookback_epoch_time(from_: Optional[datetime.datetime] = None, **kwargs):
    if from_ is None:
        from_ = datetime.datetime.today().date()
    return (from_ - relativedelta(**kwargs)).strftime("%s")


def llamaindex_textnode_to_document(textnode: TextNode):
    metadata = textnode.metadata
    metadata["char_range"] = f"{textnode.start_char_idx},{textnode.end_char_idx}"
    metadata["chunk_id"] = textnode.id_
    return Document(page_content=textnode.text, metadata=metadata)


def get_context(doc: Document, context_field: str = "window"):
    if context_field == "page_content":
        return doc.page_content
    else:
        return doc.metadata[context_field]


def get_batches(full_list: List, batch_size: int):
    for i in range(0, len(full_list), batch_size):
        yield full_list[i : i + batch_size]


def monitor_llm_response_metadata(response_metadata):
    if "stop_reason" in response_metadata:
        # Claude model
        if response_metadata["stop_reason"] != "end_turn":
            logging.warning(f"Issue with LLM response: {response_metadata}")

    if "finish_reason" in response_metadata:
        # ChatGPT model
        if response_metadata["finish_reason"] != "stop":
            logging.warning(f"Issue with LLM response: {response_metadata}")


def load_persisted_runs(
    persist_path: Path,
    refresh: bool = False,
):
    if Path(persist_path).exists() and (not refresh):
        with open(persist_path, "rb") as f:
            logging.info(f"Persisted results found, loading from {persist_path}")
            res = pickle.load(f)
        return res


def persist_data(data: Any, persist_path: Path):
    with open(persist_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
