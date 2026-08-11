---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 28px;
    padding: 60px;
  }
  h1 { font-size: 52px; color: #1a1a2e; }
  h2 { font-size: 40px; color: #1a1a2e; border-bottom: 3px solid #4a90d9; padding-bottom: 12px; }
  strong { color: #4a90d9; }
  section.lead { text-align: center; justify-content: center; }
  section.lead h1 { font-size: 64px; }
  section.lead p { font-size: 22px; color: #666; }
  ul { line-height: 1.9; }
  table { width: 100%; font-size: 24px; }
  th { background: #4a90d9; color: white; padding: 10px; }
  td { padding: 10px; border-bottom: 1px solid #eee; }
---

<!-- _class: lead -->

# blue-data-doc-source

### Internal Review
Blue Data Intelligence

---

## The Gap We're Closing

Blue Data agents can already reason over structured data —
databases, APIs, JSON feeds.

**But most enterprise knowledge lives in documents.**

> Policy manuals. Contracts. Compliance reports.
> Benefits guides. Internal knowledge bases.

Until now, agents had no way to read, search, or cite these.

---

## What We Built

A document layer for Blue Data Intelligence.

Drop in any document. Ask any question.
Get back answers with **exact sources** — which document, which section, which page.

It works with the same interface agents already use for every other data source.
**No new patterns to learn. No special integration.**

---

## What It Enables

Agents can now answer questions like:

- *"What are the eligibility requirements in this policy?"*
- *"Which clause covers termination of contract?"*
- *"What changed between last quarter's report and this one?"*

And cite exactly where the answer came from.

---

## Quick Demo

*Ingest an insurance policy PDF. Ask a question. Get ranked results with page and section.*

---

## How It Works — In Plain Terms

Three steps happen automatically when you hand it a document:

1. **Read** — extracts text from PDF or plain text files
2. **Understand structure** — detects sections and page boundaries
3. **Index** — stores searchable, AI-ready chunks in the vector store

Re-ingesting the same document is always safe — no duplicates, no manual cleanup needed.

---

## Where It Fits in Blue Data

```
Blue Data Intelligence
  ├── Databases
  ├── APIs & JSON feeds  
  ├── Graph sources
  └── Documents  ← blue-data-doc-source
```

Documents become **first-class data sources** —
same interface, same agent patterns, same infrastructure.

---

## Status

| | |
|---|---|
| **Completeness** | Ingestion, search, delete — full lifecycle covered |
| **Natural language queries** | Ask in plain English — LLM parses intent automatically |
| **File types** | PDF and plain text today |
| **Test coverage** | 111 tests, all passing |
| **Demo** | Running end-to-end with real data |



---

## The Harder Problem Ahead

Document search sounds simple. It isn't.

Tag a resume and you lose *"rebuilt the forecasting model single-handedly, cut error by 40%"* — it becomes `ml_engineer=1`. The ownership and scale disappear.

Embed it and *"led a team turnaround"* lands next to *"handled coordination"* — same vector, completely different substance.

**We've built the right foundation:** every chunk keeps its full original sentence alongside section, page, and provenance. Nothing is thrown away before reasoning happens.

**What we haven't solved yet:** ranking by evidence quality, bridging the gap between a job requirement and a resume achievement, and re-ranking results by substance rather than surface similarity.

That's the next layer.

---

## What's Next

- **Natural language queries** — `execute_query_nl("find the termination clause")` — already implemented, LLM parses intent into the right op automatically
- **MCP integration** — Claude Desktop and any AI client can search documents directly, zero extra integration work
- **More file formats** — Word, HTML, email, spreadsheets
- **Smarter search** — re-ranking results for higher answer quality and evidence alignment

---

<!-- _class: lead -->

## Thank You

*Questions?*
