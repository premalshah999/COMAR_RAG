import type { Conversation, Filters } from "../types";

/* ── Inline SVG icons ─────────────────────────────────────────────────── */
function PlusIcon() {
  return (
    <svg width={14} height={14} fill="none" stroke="currentColor" strokeWidth={2}
      strokeLinecap="round" viewBox="0 0 24 24">
      <line x1={12} y1={5} x2={12} y2={19}/><line x1={5} y1={12} x2={19} y2={12}/>
    </svg>
  );
}
function LawIcon() {
  return (
    <svg width={16} height={16} fill="none" stroke="currentColor" strokeWidth={2}
      strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
  );
}
function SunIcon() {
  return (
    <svg width={15} height={15} fill="none" stroke="currentColor" strokeWidth={2}
      strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
      <circle cx={12} cy={12} r={5}/>
      <line x1={12} y1={1} x2={12} y2={3}/><line x1={12} y1={21} x2={12} y2={23}/>
      <line x1={4.22} y1={4.22} x2={5.64} y2={5.64}/><line x1={18.36} y1={18.36} x2={19.78} y2={19.78}/>
      <line x1={1} y1={12} x2={3} y2={12}/><line x1={21} y1={12} x2={23} y2={12}/>
      <line x1={4.22} y1={19.78} x2={5.64} y2={18.36}/><line x1={18.36} y1={5.64} x2={19.78} y2={4.22}/>
    </svg>
  );
}
function MoonIcon() {
  return (
    <svg width={15} height={15} fill="none" stroke="currentColor" strokeWidth={2}
      strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
  );
}
function BookIcon() {
  return (
    <svg width={15} height={15} fill="none" stroke="currentColor" strokeWidth={2}
      strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
    </svg>
  );
}
function HomeIcon() {
  return (
    <svg width={15} height={15} fill="none" stroke="currentColor" strokeWidth={2}
      strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
      <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
      <polyline points="9 22 9 12 15 12 15 22"/>
    </svg>
  );
}
function CheckIcon() {
  return (
    <svg width={9} height={9} fill="none" stroke="white" strokeWidth={2.5}
      strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 12 12">
      <path d="M10 3L5 8.5 2 5.5"/>
    </svg>
  );
}

interface Props {
  conversations: Conversation[];
  activeId: string;
  onSelect: (id: string) => void;
  onNew: () => void;
  filters: Filters;
  onFiltersChange: (f: Filters) => void;
  onHome: () => void;
  isDark: boolean;
  onToggleTheme: () => void;
}

function TitleFilter({ filters, onChange }: { filters: Filters; onChange: (f: Filters) => void }) {
  const titles = [
    { num: "15", label: "Title 15 — Agriculture" },
    { num: "26", label: "Title 26 — Environment" },
  ];
  return (
    <>
      {titles.map(({ num, label }) => {
        const active = filters.title_num.includes(num);
        return (
          <button
            key={num}
            className={`filter-btn${active ? " active" : ""}`}
            onClick={() => {
              const next = active
                ? filters.title_num.filter((n) => n !== num)
                : [...filters.title_num, num];
              onChange({ ...filters, title_num: next });
            }}
          >
            <div className="filter-checkbox">{active && <CheckIcon />}</div>
            <span className="filter-btn-label">{label}</span>
          </button>
        );
      })}
    </>
  );
}

export function Sidebar({
  conversations, activeId, onSelect, onNew,
  filters, onFiltersChange,
  onHome, isDark, onToggleTheme,
}: Props) {
  return (
    <aside className="sidebar">
      {/* Header: brand + new chat */}
      <div className="sidebar-header">
        <div className="brand">
          <div className="brand-icon"><LawIcon /></div>
          <div>
            <div className="brand-name">COMAR</div>
            <div className="brand-sub">Maryland Regs</div>
          </div>
        </div>
        <button className="new-chat-btn" onClick={onNew}>
          <PlusIcon /> New conversation
        </button>
      </div>

      {/* Conversation history */}
      <div className="sidebar-history">
        {conversations.length === 0 ? (
          <div style={{ padding: "4px 8px", fontSize: 12, color: "var(--text-muted)" }}>
            No conversations yet
          </div>
        ) : (
          <>
            <div className="history-section-label">Recent</div>
            {conversations.map((c) => (
              <button
                key={c.id}
                className={`chat-item${c.id === activeId ? " active" : ""}`}
                onClick={() => onSelect(c.id)}
              >
                <div className="chat-item-dot" />
                <div className="chat-item-text">{c.title}</div>
              </button>
            ))}
          </>
        )}
      </div>

      {/* Filters */}
      <div className="sidebar-panel-label">Filter by Title</div>
      <TitleFilter filters={filters} onChange={onFiltersChange} />

      {/* Bottom actions */}
      <div className="sidebar-bottom">
        <button className="sidebar-action" onClick={onHome}>
          <span className="sidebar-action-icon"><HomeIcon /></span>
          <span className="sidebar-action-label">Home</span>
        </button>
        <button className="sidebar-action" onClick={onToggleTheme}>
          <span className="sidebar-action-icon">
            {isDark ? <SunIcon /> : <MoonIcon />}
          </span>
          <span className="sidebar-action-label">{isDark ? "Light mode" : "Dark mode"}</span>
        </button>
        <div className="sidebar-divider" />
        <button className="sidebar-action">
          <span className="sidebar-action-icon"><BookIcon /></span>
          <span className="sidebar-action-label">COMAR Reference</span>
        </button>
      </div>
    </aside>
  );
}
