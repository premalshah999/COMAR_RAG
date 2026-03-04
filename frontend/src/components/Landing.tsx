
/* ── SVG Icon helper ──────────────────────────────────────────────────── */
function Icon({ name, size = 16 }: { name: string; size?: number }) {
  const s = {
    width: size, height: size, fill: "none", stroke: "currentColor",
    strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const,
    viewBox: "0 0 24 24",
  };
  const paths: Record<string, React.ReactNode> = {
    law:    <svg {...s}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>,
    book:   <svg {...s}><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>,
    share:  <svg {...s}><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>,
    copy:   <svg {...s}><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>,
    thumb:  <svg {...s}><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>,
    arrow:  <svg {...s} strokeWidth={2.5}><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>,
    layers: <svg {...s}><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>,
    db:     <svg {...s}><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>,
    zap:    <svg {...s}><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>,
    home:   <svg {...s}><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>,
  };
  return paths[name] ?? null;
}

/* ── Data ─────────────────────────────────────────────────────────────── */
const PIPELINE_STEPS = [
  {
    title: "XML Ingestion & Structural Parsing",
    desc: "COMAR Titles 15 & 26 are fetched from the Maryland DSD law-xml repository on GitHub. XML files are parsed using lxml to extract regulation sections, effective dates, and cross-references, preserving the complete title → subtitle → chapter → regulation hierarchy.",
  },
  {
    title: "Semantic Chunking & Knowledge Graph",
    desc: "Each regulation is split into a primary regulation chunk and fine-grained subsection chunks with full structural metadata. A NetworkX knowledge graph (3,704 nodes, 5,762 edges) captures CONTAINS, REFERENCES, and DEFINES relationships for context-aware retrieval.",
  },
  {
    title: "BGE-M3 Hybrid Embedding & Qdrant Indexing",
    desc: "All 50,827 chunks are encoded using BAAI/BGE-M3 in a single forward pass, producing dense (1024-dim) and SPLADE sparse vectors. Both vector types are indexed in a local Qdrant instance with citation payload indices for hybrid semantic + lexical retrieval.",
  },
  {
    title: "Hybrid RRF Retrieval & Grounded Generation",
    desc: "Queries are embedded in one BGE-M3 pass. Dense and sparse results are fused via Reciprocal Rank Fusion (RRF). Exact COMAR citation patterns trigger a direct scroll lookup at rank 1. Retrieved regulatory text is passed untruncated to DeepSeek-V3 for citation-linked answers streamed token-by-token.",
  },
];

const RESEARCH_QUESTIONS = [
  {
    title: "RQ1 — Retrieval Precision",
    desc: "How accurately does BGE-M3 hybrid RRF retrieval identify the most relevant COMAR sections for natural-language queries compared to dense-only or keyword-only baselines?",
  },
  {
    title: "RQ2 — Answer Faithfulness",
    desc: "To what extent are generated answers grounded in the retrieved regulatory text versus introducing unsupported claims not present in the source regulations?",
  },
  {
    title: "RQ3 — User Accessibility",
    desc: "Does the system meaningfully reduce the time and effort required for researchers and practitioners to locate, interpret, and cite specific COMAR regulations?",
  },
];

const TEAM = [
  { initials: "JP", name: "James Purtilo",  role: "Professor, Dept. of Computer Science" },
  { initials: "PS", name: "Premal Shah",    role: "Graduate Researcher, UMD CS" },
];

const CITATION = `@misc{comarrag2025,
  title   = {COMAR Assistant: Hybrid RAG for the
             Maryland Code of Regulations},
  author  = {Shah, Premal and Purtilo, James and
             {University of Maryland CS}},
  year    = {2025},
  note    = {Research prototype. COMAR Title 15
             (Agriculture) and Title 26 (Environment).
             Not for commercial use.},
  url     = {https://cs.umd.edu}
}`;

interface Props {
  onEnter: () => void;
}

export function Landing({ onEnter }: Props) {
  return (
    <div className="landing page-enter">

      {/* ── Top bar ── */}
      <div className="land-topbar">
        <div className="land-inst-row">
          <div className="land-inst-logo">
            <div className="land-inst-icon">
              <Icon name="law" size={14} />
            </div>
            <span className="land-inst-name">COMAR Assistant</span>
          </div>
          <div className="land-topbar-sep" />
          <span className="land-inst-dept">University of Maryland · Dept. of Computer Science</span>
        </div>
        <div className="land-topbar-right">
          <button className="land-topbar-link">Abstract</button>
          <button className="land-topbar-link">Architecture</button>
          <button className="land-topbar-link">Team</button>
          <button className="land-open-btn" onClick={onEnter}>
            <Icon name="arrow" size={12} /> Open App
          </button>
        </div>
      </div>

      {/* ── Body ── */}
      <div className="land-body">

        {/* Paper title block */}
        <div className="land-paper-header">
          <div className="land-venue-tag">
            <div className="land-venue-dot" />
            Research Prototype · 2025
            <div className="land-venue-dot" />
            University of Maryland, College Park
          </div>

          <h1 className="land-title">
            COMAR Assistant: Hybrid Retrieval-Augmented<br />
            Generation for Maryland Regulations
          </h1>

          <div className="land-authors">
            <span className="land-author">Premal Shah</span>
            <span className="land-author-sep">·</span>
            <span className="land-author">James Purtilo</span>
          </div>

          <div className="land-affil">
            Department of Computer Science · University of Maryland, College Park
          </div>

          <div className="land-paper-links">
            <button className="land-pill land-pill-primary" onClick={onEnter}>
              <Icon name="law" size={12} /> Try the Demo
            </button>
            <button className="land-pill">
              <Icon name="book" size={12} /> Paper (forthcoming)
            </button>
            <button className="land-pill">
              <Icon name="share" size={12} /> GitHub
            </button>
            <button className="land-pill">
              <Icon name="db" size={12} /> Dataset
            </button>
          </div>
        </div>

        {/* Two-column content */}
        <div className="land-cols">

          {/* Left: main content */}
          <div>
            {/* Abstract */}
            <div className="land-section">
              <div className="land-section-heading">Abstract</div>
              <p className="land-abstract">
                Regulatory text is notoriously difficult to navigate — dense, hierarchical, and
                distributed across thousands of sections. This project explores whether hybrid
                retrieval-augmented generation (RAG) can meaningfully improve access to the Maryland
                Code of Regulations (COMAR) for researchers, practitioners, and the general public.
              </p>
              <p className="land-abstract" style={{ marginTop: 12 }}>
                We built a system that ingests the full text of COMAR Titles 15 (Agriculture) and
                26 (Environment) from the Maryland Division of State Documents, indexes 50,827 chunks
                using BGE-M3 dense and sparse embeddings in Qdrant, and answers user queries by
                retrieving the most relevant regulatory passages via Reciprocal Rank Fusion before
                generating grounded, citation-linked answers with DeepSeek-V3. All answers are
                traceable to their source regulation — minimizing hallucination and enabling
                independent verification.
              </p>
            </div>

            {/* System Architecture */}
            <div className="land-section">
              <div className="land-section-heading">System Architecture</div>
              <div className="land-pipeline">
                {PIPELINE_STEPS.map((step, i) => (
                  <div className="land-pipeline-step" key={i}>
                    <div className="land-step-num">0{i + 1}</div>
                    <div className="land-step-content">
                      <div className="land-step-title">{step.title}</div>
                      <div className="land-step-desc">{step.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Research Questions */}
            <div className="land-section">
              <div className="land-section-heading">Research Questions</div>
              <div className="land-pipeline">
                {RESEARCH_QUESTIONS.map((rq, i) => (
                  <div className="land-pipeline-step" key={i}>
                    <div className="land-step-content">
                      <div className="land-step-title">{rq.title}</div>
                      <div className="land-step-desc">{rq.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Citation */}
            <div className="land-section">
              <div className="land-section-heading">Citation</div>
              <div className="land-citation-block">{CITATION}</div>
            </div>
          </div>

          {/* Right: metadata sidebar */}
          <div>
            <div className="land-meta-card">
              <div className="land-meta-card-header">Project Details</div>
              <div className="land-meta-rows">
                {[
                  ["Status",       "Active · 2025"],
                  ["Type",         "Research Prototype"],
                  ["Institution",  "UMD, College Park"],
                  ["Department",   "Dept. of CS"],
                  ["Course",       "CMSC 607"],
                  ["License",      "Non-commercial"],
                ].map(([k, v]) => (
                  <div className="land-meta-row" key={k}>
                    <span className="land-meta-key">{k}</span>
                    <span className="land-meta-val">{v}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="land-meta-card">
              <div className="land-meta-card-header">Corpus</div>
              <div className="land-meta-rows">
                {[
                  ["Source",       "MD DSD law-xml"],
                  ["Titles",       "15 & 26"],
                  ["Regulations",  "3,309"],
                  ["Chunks",       "50,827"],
                  ["Graph nodes",  "3,704"],
                  ["Graph edges",  "5,762"],
                ].map(([k, v]) => (
                  <div className="land-meta-row" key={k}>
                    <span className="land-meta-key">{k}</span>
                    <span className="land-meta-val">{v}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="land-meta-card">
              <div className="land-meta-card-header">Stack</div>
              <div className="land-meta-rows">
                {[
                  ["Embeddings",   "BAAI/BGE-M3"],
                  ["Vector DB",    "Qdrant"],
                  ["Fusion",       "Hybrid RRF"],
                  ["Graph",        "NetworkX"],
                  ["LLM",          "DeepSeek-V3"],
                  ["Backend",      "Python / FastAPI"],
                  ["Frontend",     "React + Vite"],
                ].map(([k, v]) => (
                  <div className="land-meta-row" key={k}>
                    <span className="land-meta-key">{k}</span>
                    <span className="land-meta-val">{v}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="land-meta-card">
              <div className="land-meta-card-header">Team</div>
              <div>
                {TEAM.map((m) => (
                  <div className="land-team-member" key={m.initials}>
                    <div className="land-team-avatar">{m.initials}</div>
                    <div className="land-team-info">
                      <div className="land-team-name">{m.name}</div>
                      <div className="land-team-role">{m.role}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="land-footer">
        <span className="land-footer-copy">
          University of Maryland · Dept. of Computer Science · Research use only · 2025
        </span>
        <div className="land-footer-links">
          <a className="land-footer-link" href="https://github.com" target="_blank" rel="noopener noreferrer">GitHub</a>
          <button className="land-footer-link">Paper</button>
          <a className="land-footer-link" href="https://regs.maryland.gov" target="_blank" rel="noopener noreferrer">COMAR</a>
        </div>
      </footer>
    </div>
  );
}
