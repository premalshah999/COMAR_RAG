import { Scale } from "lucide-react";
import type { HealthStatus, Theme } from "../types";
import { ThemeToggle } from "./ThemeToggle";

interface Props {
  theme: Theme;
  onToggle: () => void;
  health: HealthStatus | null;
}

function StatusDot({ health }: { health: HealthStatus | null }) {
  if (!health || health.status === "loading") {
    return (
      <span className="flex items-center gap-1.5 text-xs dark:text-azure-400/60 text-azure-600/60">
        <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse-slow" />
        connecting…
      </span>
    );
  }
  const ok = health.status === "ok";
  const pct = health.qdrant_points
    ? Math.round((health.qdrant_points / 52528) * 100)
    : 0;

  return (
    <span className="flex items-center gap-1.5 text-xs">
      <span
        className={`w-1.5 h-1.5 rounded-full ${
          ok ? "bg-emerald-500" : "bg-amber-400"
        }`}
      />
      <span className="dark:text-azure-400/80 text-azure-600/80 font-mono">
        {health.qdrant_points.toLocaleString()} vectors
        {pct < 100 && ` · ${pct}%`}
      </span>
      {health.llm_ready && (
        <span className="dark:text-azure-400/60 text-azure-500/60">· LLM ready</span>
      )}
    </span>
  );
}

export function Header({ theme, onToggle, health }: Props) {
  return (
    <>
      <header
        className="
          h-14 flex-shrink-0 flex items-center justify-between px-5
          dark:border-b dark:border-azure-900/40
          border-b border-azure-200/60
          dark:bg-ink-900/80 bg-white/80 backdrop-blur-md
          z-20 relative
        "
      >
        {/* Brand */}
        <div className="flex items-center gap-2.5">
          <div
            className="
              w-8 h-8 rounded-xl flex items-center justify-center
              dark:bg-azure-700 bg-azure-600
              shadow-glow-sm
            "
          >
            <Scale size={15} className="text-white" strokeWidth={2.5} />
          </div>
          <div className="flex flex-col -space-y-0.5">
            <span className="text-sm font-semibold tracking-tight dark:text-white text-ink-900">
              COMAR Assistant
            </span>
            <span className="text-[10px] font-mono dark:text-azure-500 text-azure-600 uppercase tracking-wider">
              Maryland Regulations
            </span>
          </div>
        </div>

        {/* Right controls */}
        <div className="flex items-center gap-4">
          <StatusDot health={health} />
          <ThemeToggle theme={theme} onToggle={onToggle} />
        </div>
      </header>

      {/* Legal disclaimer strip */}
      <div
        className="
          flex-shrink-0 w-full text-center text-[10px] py-1 px-4
          dark:bg-amber-950/40 bg-amber-50
          dark:text-amber-400/80 text-amber-700
          dark:border-b dark:border-amber-900/30 border-b border-amber-200
        "
      >
        ⚖️ For informational research only — not legal advice. Verify all
        information with official COMAR at{" "}
        <a
          href="https://regs.maryland.gov"
          target="_blank"
          rel="noopener noreferrer"
          className="underline underline-offset-2 hover:opacity-80"
        >
          regs.maryland.gov
        </a>{" "}
        before taking any regulatory action.
      </div>
    </>
  );
}
