import { useState } from "react";
import type { Source } from "../types";

interface Props {
  source: Source;
  rank: number;
}

function ChevronDown() {
  return (
    <svg width={11} height={11} fill="none" stroke="currentColor" strokeWidth={2}
      strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
      <polyline points="6 9 12 15 18 9"/>
    </svg>
  );
}
function ChevronUp() {
  return (
    <svg width={11} height={11} fill="none" stroke="currentColor" strokeWidth={2}
      strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
      <polyline points="18 15 12 9 6 15"/>
    </svg>
  );
}

export function CitationCard({ source, rank }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="citation-card">
      <div className="citation-card-header" onClick={() => setExpanded((e) => !e)}>
        {/* Rank bubble */}
        <div className="citation-rank">{rank}</div>

        {/* Meta */}
        <div className="citation-meta">
          <div className="citation-code">
            {source.citation}
            <span className="citation-type-chip">{source.chunk_type}</span>
          </div>
          <div className="citation-name">{source.regulation_name}</div>
          <div className="citation-breadcrumb">
            {source.subtitle_name} › {source.chapter_name}
          </div>
          {source.effective_date && (
            <div className="citation-breadcrumb" style={{ marginTop: 2 }}>
              Effective: {source.effective_date}
            </div>
          )}
        </div>

        {/* Chevron */}
        <div className="citation-chevron">
          {expanded ? <ChevronUp /> : <ChevronDown />}
        </div>
      </div>

      {/* Expanded body */}
      {expanded && (
        <div className="citation-card-body">
          {source.text_snippet}
        </div>
      )}
    </div>
  );
}
