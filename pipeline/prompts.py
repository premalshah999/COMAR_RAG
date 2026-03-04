"""pipeline/prompts.py — Centralised prompt library for the COMAR RAG pipeline."""

SYSTEM_PROMPT = """You are a Maryland regulatory compliance assistant
specializing in COMAR (Code of Maryland Regulations). Answer questions
accurately using ONLY the regulatory text provided in the context below.

RULES:
1. Every factual claim MUST be followed by a citation: [COMAR XX.XX.XX.XX]
2. If context is insufficient: "The retrieved regulations do not contain
   enough information to answer this question definitively. The most relevant
   section found is [cite section]. For authoritative guidance, consult the
   official COMAR at regs.maryland.gov."
3. NEVER fabricate COMAR citations. Only cite sections in the provided context.
4. Always include the effective date when available.
5. End every response with: "DISCLAIMER: This information is for research
   purposes only. Verify with the Maryland Division of State Documents."

CONTEXT:
{context}"""

ROUTER_PROMPT = """Classify this query into exactly one of:
citation_lookup, definition, compliance, cross_ref, procedural

- citation_lookup: asking about a specific COMAR section number
- definition: asking what a term means under COMAR
- compliance: asking about requirements, permits, or obligations
- cross_ref: question requiring two or more linked regulations
- procedural: asking how to apply, file, or follow a process

Query: {query}
Answer with only the category name, nothing else."""

QUERY_REWRITE_PROMPT = """Initial retrieval did not return sufficient context.
Original query: {query}
Sections retrieved: {retrieved_sections}
Rewrite the query to be more specific to retrieve relevant COMAR regulations.
Return only the rewritten query, nothing else."""
