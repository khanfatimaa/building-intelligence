import json
import time
import uuid

from fastapi import FastAPI, Request
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
from .service import run_pattern
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
def query(request: QueryRequest) -> QueryResult:
    return run_pattern(request)


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


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
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
    text = run_pattern(query_request).message

    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    if not request.stream:
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

    def frames():
        yield _sse(chat_id, created, request.model, {"role": "assistant", "content": ""}, None)
        yield _sse(chat_id, created, request.model, {"content": text}, None)
        yield _sse(chat_id, created, request.model, {}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream")
