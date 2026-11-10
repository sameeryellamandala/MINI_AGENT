# Signal Search — Design

## 1. Challenge interpretation

The agent must answer quickly, get factual questions right, and remember what the user asked and when. It may have one tool: web search.

The strict boundary in this implementation is simple and auditable:

```python
tools=[{"type": "browser_search"}]
```

No other model-callable tool is defined. LangGraph, SQLite, and Streamlit are application infrastructure, not capabilities exposed to the model.

## 2. Goals

1. **Fast answers:** use Groq inference, a compact prompt, low reasoning effort, and a bounded history window.
2. **Correct answers:** search current facts, prefer authoritative sources, cross-check important claims, and expose citations.
3. **Durable memory:** restore conversations after a process restart and attach a timezone-aware timestamp to every question.
4. **Small operational surface:** run with one API key and one command; require no database server or separate search account.

## 3. Architecture

```text
┌──────────────────────┐
│ Streamlit            │  chat, thread picker, timestamps
└──────────┬───────────┘
           │ question + asked_at
           ▼
┌──────────────────────┐
│ LangGraph            │  one node; persistent thread state
└──────┬─────────┬─────┘
       │         └──────────────► SQLite
       │                          checkpoints + thread index
       ▼
┌──────────────────────┐
│ Groq GPT-OSS         │  choose whether search is needed
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ browser_search       │  only model-callable tool
└──────────────────────┘
```

The graph deliberately contains one `answer` node. More nodes would add ceremony without improving a one-tool loop. Groq performs hosted browser-search orchestration; LangGraph provides durable state and thread isolation.

## 4. Search choice

Groq's built-in `browser_search` is used instead of Wikipedia or an unofficial DuckDuckGo wrapper.

- Wikipedia covers one corpus and cannot reliably answer breaking or niche questions.
- Unofficial search wrappers can be throttled or change without notice.
- Groq browser search covers the live web, requires no second API key, and is supported by the selected GPT-OSS models.

`tool_choice="auto"` avoids pointless searches for greetings and memory questions while preserving the one-tool boundary. The system prompt requires search for current, niche, time-sensitive, or externally verifiable facts.

## 5. Memory and time

Each user message is stored as raw state:

```json
{
  "role": "user",
  "content": "What changed in AI this week?",
  "asked_at": "2026-11-10T20:30:00+05:30"
}
```

LangGraph's SQLite checkpointer saves the full state under a `thread_id`. A small SQLite thread index powers the sidebar. Before inference, timestamps are included in the model-visible history, enabling questions such as “What did I ask before, and when?”

The last 24 messages are sent to the model. Full state remains stored; bounding model context prevents latency and cost from growing without limit.

## 6. Accuracy and safety

- Search is requested for claims likely to have changed.
- The prompt prioritizes primary and authoritative sources.
- Important claims should be checked across sources.
- Search results are treated as untrusted evidence, reducing prompt-injection risk.
- The agent is instructed not to invent citations and to state uncertainty.
- The UI renders Markdown links so sources remain inspectable.

No search system guarantees truth. The design makes uncertainty visible and gives the user direct access to the evidence.

## 7. Latency decisions

- Groq-hosted `openai/gpt-oss-20b` is the default.
- Reasoning effort is `low`, which Groq recommends as a speed/quality balance for browser search.
- Temperature is `0.2` for stable factual answers.
- Maximum output is bounded at 1,800 tokens.
- Search is automatic rather than mandatory for every conversational turn.
- The UI shows immediate search status while the hosted tool runs.

The optional 120B model improves difficult-answer quality but may consume free-tier limits faster.

## 8. Failure handling

- Missing API key: show setup instructions without crashing.
- Free-tier rate limit: show a specific retry message.
- Other provider errors: present a safe generic message rather than leaking credentials.
- Empty provider response: return an explicit retry request.
- SQLite state: stored locally and ignored by Git.

## 9. Testing strategy

Automated tests use a fake Groq client and verify:

1. The request exposes exactly one tool and it is `browser_search`.
2. Questions and timestamps survive in thread state.
3. Follow-up requests receive prior timestamped history.
4. Recently updated conversations appear first.

Manual acceptance test:

1. Ask a current factual question and inspect the source links.
2. Ask a follow-up using a pronoun to confirm conversational context.
3. Ask “What was my first question and when did I ask it?”
4. Restart Streamlit and reopen the thread from the sidebar.

## 10. Tradeoffs and next steps

SQLite is the right submission default: zero setup, durable, and easy to inspect. A multi-instance production deployment would replace it with PostgreSQL. The deliberately small graph could later add summarization for extremely long chats, but that is excluded here to keep the challenge solution focused and fast.

