import praw
from typing import Callable, List, Optional
from langchain_core.documents import Document
import re
from functools import cache
from disraptors.rag.utils import (
    load_config,
    REGEX_CASE_SENSITIVE_LEVELS,
)
import tqdm
from functools import reduce


@cache
def extract_comments(post, only_toplevel_comments: bool = True):
    post.comments.replace_more(limit=1)
    if only_toplevel_comments:
        return post.comments
    else:
        all_comments = post.comments.list()
        return [c for c in all_comments if not isinstance(c, praw.models.MoreComments)]


def is_phrase_in_post(
    post: str,
    phrase: str,
    case_sensitivity: str = "strict",
    only_toplevel_comments: bool = True,
):
    assert (
        case_sensitivity in REGEX_CASE_SENSITIVE_LEVELS
    ), f"case_sensitivity must be either {REGEX_CASE_SENSITIVE_LEVELS}"

    flag = re.IGNORECASE if case_sensitivity == "ignore" else 0

    if case_sensitivity == "title":
        phrase = f"{phrase}|{phrase.title()}"

    for key in ["title", "selftext"]:
        if re.search(r"\b{}\b".format(phrase), getattr(post, key, ""), flags=flag):
            return True

    for comment in extract_comments(post, only_toplevel_comments=only_toplevel_comments):
        if re.search(r"\b{}\b".format(phrase), comment.body, flags=flag):
            return True

    return False


class Reddit:
    """Reddit API client. Only match in posts is enabled, no comment search.
    API doc: https://www.reddit.com/dev/api#GET_search"""

    def __init__(self):
        config = load_config("secret.config")["reddit"]
        self.client = praw.Reddit(
            client_id=config["client_id"],
            client_secret=config["client_secret"],
            user_agent=config["user_agent"],
        )

    def search_subreddit(
        self,
        subreddit: str,
        query,
        sort="relevance",
        time_filter="all",
        limit=10,
        phrase_check: Optional[str] = None,
        case_sensitivity: str = "strict",
        only_toplevel_comments: bool = True,
    ) -> List:

        subreddit = self.client.subreddit(subreddit)
        response = list(subreddit.search(query, sort=sort, time_filter=time_filter, limit=limit))

        if phrase_check is None:
            return response

        # Check if a given phrase exists in the post
        posts = []

        for post in tqdm.tqdm(response, desc="Filtering post by phrase", total=len(response)):
            # confirm if provided phrase exists in post
            if is_phrase_in_post(
                post=post,
                phrase=phrase_check,
                case_sensitivity=case_sensitivity,
                only_toplevel_comments=only_toplevel_comments,
            ):
                posts.append(post)

        return posts

    def get_hot_posts(self, subreddit: str, limit: int = 50):
        subreddit = self.client.subreddit(subreddit)
        return list(subreddit.hot(limit=limit))

    def get_post_from_id(self, post_id):
        return self.client.submission(id=post_id)


# Formatters


class RedditComment:
    def __init__(self, comment):
        self.comment = comment

    def to_json(self):
        comment_json = {
            "comment_id": self.comment.id,
            "is_comment_submitter": self.comment.is_submitter,
            "comment_created_utc": int(self.comment.created_utc),
            "comment_body": self.comment.body,
            "post_id": self.comment.submission.id,
            "subreddit": self.comment.subreddit.display_name,
        }
        return comment_json


class RedditPost:
    def __init__(self, post, include_comments: bool = True, only_toplevel_comments: bool = True):
        self.post = post
        self.inclue_comments = include_comments
        self.only_toplevel_comments = only_toplevel_comments

    def to_json(self):
        post_json = {
            "subreddit": self.post.subreddit.display_name,
            "post_id": self.post.id,
            "post_author": self.post.author.name,
            "post_title": self.post.title,
            "post_body": self.post.selftext,
            "post_created_utc": int(self.post.created_utc),
            "is_post_oc": self.post.is_original_content,
            "is_post_video": self.post.is_video,
            "post_upvote_count": self.post.ups,
            "post_downvote_count": self.post.downs,
            "subreddit_members": self.post.subreddit_subscribers,
            "url": self.post.url,
        }

        if self.include_comments:
            comments = extract_comments(post=self.post, only_toplevel_comments=self.only_toplevel_comments)
            post_json["comments"] = [RedditComment(comment).to_json() for comment in comments]
            post_json["comment_count"] = len(comments)
            post_json["only_toplevel_comments"] = self.only_toplevel_comments

        return post_json

    def to_document(self) -> List[Document]:
        doc = Document(
            page_content=(f"{self.post.title}\n{self.post.selftext}"),
            metadata={
                "subreddit": self.post.subreddit.display_name,
                "post_id": self.post.id,
                "post_author": self.post.author.name,
                "post_title": self.post.title,
                "post_created_utc": int(self.post.created_utc),
                "is_post_oc": self.post.is_original_content,
                "is_post_video": self.post.is_video,
                "post_upvote_count": self.post.ups,
                "post_downvote_count": self.post.downs,
                "subreddit_members": self.post.subreddit_subscribers,
                "url": "https://www.reddit.com" + self.post.permalink,
            },
        )

        if self.inclue_comments:
            comments = extract_comments(post=self.post, only_toplevel_comments=self.only_toplevel_comments)
            doc.metadata["comment_count"] = len(comments)
            doc.metadata["only_toplevel_comments"] = self.only_toplevel_comments

            comment_docs = []
            for comment in comments:
                comment_metadata = doc.metadata.copy()
                comment_metadata.update(
                    {
                        "is_comment": True,
                        "comment_id": comment.id,
                        "is_comment_submitter": comment.is_submitter,
                        "comment_created_utc": int(comment.created_utc),
                        "comment_parent_id": comment.parent_id,
                        "comment_depth": comment.depth,
                        "comment_upvote_count": comment.ups,
                        "comment_downvote_count": comment.downs,
                    }
                )
                comment_docs.append(
                    Document(
                        page_content=f"({'post author' if comment.is_submitter else 'vistor'}) {comment.body}",
                        metadata=comment_metadata,
                    )
                )
            return [doc] + comment_docs
        else:
            return [doc]


def format_search_results(
    posts,
    to: str = "json",
    include_comments: bool = True,
    only_toplevel_comments: bool = True,
):
    if to == "json":
        return list(
            map(
                lambda post: RedditPost(
                    post=post,
                    include_comments=include_comments,
                    only_toplevel_comments=only_toplevel_comments,
                ).to_json(),
                posts,
            )
        )
    elif to == "document":
        list_of_list = map(
            lambda post: RedditPost(
                post=post,
                include_comments=include_comments,
                only_toplevel_comments=only_toplevel_comments,
            ).to_document(),
            posts,
        )
        return list(reduce(lambda x, y: x + y, list_of_list))
    else:
        raise ValueError(f"Invalid format {to}. Currently only support 'json' and 'document'")


def get_permalink(post: Document):
    return post.metadata["url"] + post.metadata.get("comment_id", "")
