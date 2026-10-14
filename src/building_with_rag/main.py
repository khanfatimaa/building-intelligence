import hmac
import itertools
import json
import time
import uuid

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .contracts import (
    ChatCompletionRequest,
    ModelCard,
    ModelList,
    OpenAIError,
    OpenAIErrorBody,
    QueryRequest,
    QueryResult,
    SemanticFilters,
)
from .registry import BY_MODEL_ID, MODES
from .service import run_pattern, run_pattern_events
from .settings import get_settings

app = FastAPI(title="Building with RAG")


def _openai_error(status: int, message: str, code: str | None = None) -> JSONResponse:
    body = OpenAIError(
        error=OpenAIErrorBody(message=message, type="invalid_request_error", code=code)
    )
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    if request.url.path.startswith("/v1/chat/"):
        first = errors[0] if errors else {"loc": [], "msg": ""}
        loc = ".".join(str(p) for p in first["loc"])
        return _openai_error(400, f"Invalid request: {loc}: {first['msg']}")
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResult)
def query(request: QueryRequest):
    result = run_pattern(request, answer_generation=True)
    if result.status == "unavailable":
        return JSONResponse(status_code=503, content=result.model_dump())
    return result


@app.get("/v1/models", response_model=ModelList)
def models() -> ModelList:
    return ModelList(data=[ModelCard(id=m.model_id) for m in MODES])


def _sse(chunk_id: str, created: int, model: str, delta: dict, finish: str | None) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _authorized(authorization: str | None) -> bool:
    """CAPSTONE_API_KEY gates chat only when it is set; an empty key leaves chat open."""
    expected = get_settings().capstone_api_key
    if not expected:
        return True
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    return hmac.compare_digest(supplied.encode(), expected.encode())


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest, authorization: str | None = Header(None)):
    if not _authorized(authorization):
        return _openai_error(401, "Invalid or missing API key.", "invalid_api_key")
    mode = BY_MODEL_ID.get(request.model)
    if mode is None:
        return _openai_error(404, f"The model '{request.model}' does not exist.", "model_not_found")
    question = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
    if not question.strip() or len(question) > 4000:
        return _openai_error(400, "A user message of 1-4,000 characters is required.")

    opts = request.rag_options
    filters = opts.filters if opts else None
    if opts and filters is None and (opts.act or opts.status or opts.access_level):
        filters = SemanticFilters(act=opts.act, status=opts.status, access_level=opts.access_level)
    query_request = QueryRequest(
        question=question,
        pattern=opts.pattern if opts and opts.pattern else mode.pattern,
        caller_id=get_settings().webui_demo_caller_id,
        filters=filters,
        limit=opts.limit if opts and opts.limit else 5,
        generate_answer=True,
        required_acts=opts.required_acts if opts else None,
        chapter=opts.chapter if opts else None,
    )
    # The same operation as /v1/query. Retrieval and context assembly finish inside it before the
    # first text event; only answer writing is streamed.
    events = run_pattern_events(query_request, answer_generation=True, stream=request.stream)

    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    if not request.stream:
        text = "".join(value for kind, value in events if kind == "text")
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    first = next(events, None)  # runs retrieval and context assembly before the stream opens

    def frames():
        yield _sse(chat_id, created, request.model, {"role": "assistant", "content": ""}, None)
        try:
            pending = [first] if first else []
            for kind, value in itertools.chain(pending, events):
                if kind == "text" and value:
                    yield _sse(chat_id, created, request.model, {"content": value}, None)
        except Exception:  # noqa: BLE001
            # Never leave the stream open or leak details: say so, then close normally.
            note = "\n\n**ERROR — the answer stream stopped unexpectedly; the text above is incomplete.**"
            yield _sse(chat_id, created, request.model, {"content": note}, None)
        yield _sse(chat_id, created, request.model, {}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream")
