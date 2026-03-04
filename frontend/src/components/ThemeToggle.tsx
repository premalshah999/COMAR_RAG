import { Moon, Sun } from "lucide-react";
import type { Theme } from "../types";

interface Props {
  theme: Theme;
  onToggle: () => void;
}

export function ThemeToggle({ theme, onToggle }: Props) {
  return (
    <button
      onClick={onToggle}
      className="
        relative w-9 h-9 rounded-xl flex items-center justify-center
        focus-ring transition-all duration-200
        dark:bg-ink-700 dark:hover:bg-ink-600 dark:text-azure-300 dark:hover:text-azure-200
        bg-azure-100 hover:bg-azure-200 text-azure-700 hover:text-azure-800
      "
      aria-label="Toggle theme"
    >
      <span className="sr-only">Toggle {theme === "dark" ? "light" : "dark"} mode</span>
      {theme === "dark" ? (
        <Sun size={16} className="transition-transform duration-200 hover:rotate-12" />
      ) : (
        <Moon size={16} className="transition-transform duration-200 hover:-rotate-12" />
      )}
    </button>
  );
}
