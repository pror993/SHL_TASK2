import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import generate_recommendation
from .catalog import CatalogMeta
from .config import RERANKER_ENABLED
from .qdrant import get_qdrant_client
from .planner import plan_request
from .retrieval import (
    merge_many_results,
    pipeline_constraint,
    pipeline_jd,
    preload_reranker,
    rerank_results,
)
from .validator import init_validator, validate_recommendations

ROOT_DIR = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT_DIR / "data" / "catalog.json"
CHAT_TIMEOUT_SECONDS = 25


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )


_configure_logging()
logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup: loading catalog meta from %s", CATALOG_PATH)
    catalog_meta = CatalogMeta(str(CATALOG_PATH))
    app.state.catalog_meta = catalog_meta
    init_validator(catalog_meta)
    logger.info(
        "startup: catalog loaded job_levels=%s keys=%s languages=%s",
        len(catalog_meta.job_levels),
        len(catalog_meta.keys),
        len(catalog_meta.languages),
    )

    logger.info("startup: connecting to qdrant")
    client = get_qdrant_client()
    client.get_collections()
    logger.info("startup: qdrant connection ok")

    if RERANKER_ENABLED:
        logger.info("startup: preloading reranker")
        preload_reranker()
        logger.info("startup: reranker ready")

    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    messages = request.messages
    turns_used = len(messages)
    turns_remaining = max(0, 8 - turns_used)
    logger.info("chat start messages=%s turns_remaining=%s", turns_used, turns_remaining)

    fallback_reply = (
        "Sorry, I ran out of time while preparing the response. "
        "Please try again."
    )
    partial_response = ChatResponse(
        reply=fallback_reply,
        recommendations=[],
        end_of_conversation=False,
    )

    async def _handle() -> ChatResponse:
        nonlocal partial_response

        catalog_meta = app.state.catalog_meta
        history_payload = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        planner_output = await plan_request(history_payload)
        logger.info(
            "chat planner intent=%s subqueries=%s",
            planner_output.intent,
            len(planner_output.subqueries),
        )
        logger.info(
            "chat planner raw_seniority=%s raw_test_preference=%s adaptive=%s compare_targets=%s",
            planner_output.raw_seniority,
            planner_output.raw_test_preference,
            planner_output.adaptive,
            planner_output.compare_targets,
        )

        requirement_summary = " ".join(
            message.content.strip()
            for message in messages
            if message.role == "user" and message.content.strip()
        ).strip()
        planner_subqueries = planner_output.subqueries or (
            [requirement_summary] if requirement_summary else []
        )

        if planner_output.intent == "compare" and planner_output.compare_targets:
            query_results = await asyncio.gather(
                *[
                    pipeline_jd(target_name, limit=5)
                    for target_name in planner_output.compare_targets
                ]
            )
            logger.info(
                "chat compare retrieval compare_targets=%s query_results=%s",
                len(planner_output.compare_targets),
                len(query_results),
            )
            merged = merge_many_results(list(query_results), limit=20)
        else:
            matched_job_levels = (
                catalog_meta.match_job_levels(planner_output.raw_seniority)
                if planner_output.raw_seniority
                else []
            )
            matched_keys = (
                catalog_meta.match_keys(planner_output.raw_test_preference)
                if planner_output.raw_test_preference
                else []
            )
            constraint_results, *query_results = await asyncio.gather(
                pipeline_constraint(
                    requirement_summary,
                    matched_job_levels,
                    matched_keys,
                    planner_output.adaptive,
                    limit=20,
                ),
                *[
                    pipeline_jd(subquery, limit=20)
                    for subquery in planner_subqueries
                ],
            )
            logger.info(
                "chat retrieval constraint_results=%s query_results=%s",
                len(constraint_results),
                len(query_results),
            )
            merged = merge_many_results([constraint_results, *query_results], limit=20)
        logger.info("chat merged=%s", len(merged))
        if RERANKER_ENABLED and merged:
            rerank_query = planner_output.jd_text or requirement_summary
            retrieved_items = rerank_results(rerank_query, merged[:20], top_k=10)
        else:
            retrieved_items = merged[:20]
        logger.info("chat retrieved_items=%s", len(retrieved_items))

        agent_data = await generate_recommendation(
            history_payload, retrieved_items, planner_output.intent
        )

        intent = str(agent_data.get("intent", "recommend")).strip().lower()
        reply = str(agent_data.get("reply", "")).strip() or fallback_reply
        recommendations = agent_data.get("recommendations", []) or []

        recommendations = validate_recommendations(recommendations, intent)
        end_of_conversation = bool(agent_data.get("end_of_conversation", False))
        logger.info(
            "chat response recs=%s end=%s",
            len(recommendations),
            end_of_conversation,
        )

        response = ChatResponse(
            reply=reply,
            recommendations=[Recommendation(**rec) for rec in recommendations],
            end_of_conversation=end_of_conversation,
        )
        partial_response = response
        return response

    try:
        return await asyncio.wait_for(_handle(), timeout=CHAT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("chat timeout returning partial response")
        return partial_response
