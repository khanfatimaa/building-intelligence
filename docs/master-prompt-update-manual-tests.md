Master prompt — Update curl manual tests for one story
Paste this prompt into your coding assistant from the capstone repository. Replace <STORY_NUMBER> with the implemented story number, for example 1.1, 1.4, 2.2, or 3.4.

The implemented story is Story <STORY_NUMBER>. Update docs/manual-tests.md with a short manual-test section for this story. Change only docs/manual-tests.md; do not implement or change product code, configuration, tests, data, or story files.

First read the implemented story in docs/stories/, its actual routes and request/response models, and the current docs/manual-tests.md. If no story exactly matches <STORY_NUMBER>, stop and report that. Use the current code as the authority; do not invent endpoints, payload fields, model IDs, response fields, or expected behavior.

Create or replace only the section headed ## Story <STORY_NUMBER> — .... Preserve every other story section. Keep this section short:

One sentence: What it adds.
One short prerequisite note only when needed, such as starting the API or setting a named .env variable. Never include a secret value.
A code block containing only the curl commands needed to prove the story works. Every API curl command must use the real method, URL, headers, JSON payload, and contract fields. Use http://127.0.0.1:8000 unless the implemented code documents another local address.
One brief expected-result line after each command or command group. State the important status/result field, selected pattern/model, and the expected answer, source, error, no-match, clarification, or draft behavior.
For a story with an HTTP endpoint, include:

one successful curl command;
one relevant failure, empty-result, or edge-case curl command;
a curl command for each new selectable RAG mode it adds.
For a chat or streaming story, include a real OpenAI-compatible /v1/chat/completions curl request. Use "stream": true when streaming is implemented and state that the response ends with data: [DONE]. Do not document Open WebUI steps.

For an ingestion, MongoDB, or index story with no HTTP endpoint, include only its real existing CLI command(s), not invented curl commands. Do not add a Postman, Open WebUI, automated-test, database-inspection, or production-readiness checklist.

After editing, ensure the story appears exactly once, every code fence is closed, and each command refers to a real implemented route or module. Report the section heading you updated.