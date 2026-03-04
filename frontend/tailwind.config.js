/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      colors: {
        // Dark-mode palette
        ink: {
          950: "#000000",
          900: "#08090e",
          800: "#0e1018",
          700: "#151722",
          600: "#1c1f2e",
          500: "#252840",
        },
        // Accent blues
        azure: {
          950: "#0a1628",
          900: "#0f2040",
          800: "#163060",
          700: "#1d4ed8",
          600: "#2563eb",
          500: "#3b82f6",
          400: "#60a5fa",
          300: "#93c5fd",
          200: "#bfdbfe",
          100: "#dbeafe",
          50:  "#eff6ff",
        },
      },
      animation: {
        "fade-up":    "fadeUp 0.3s ease-out",
        "fade-in":    "fadeIn 0.2s ease-out",
        "pulse-slow": "pulse 2.5s cubic-bezier(0.4,0,0.6,1) infinite",
        "blink":      "blink 1s step-end infinite",
      },
      keyframes: {
        fadeUp: {
          "0%":   { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "0%":   { opacity: "0" },
          "100%": { opacity: "1" },
        },
        blink: {
          "0%, 100%": { opacity: "1" },
          "50%":      { opacity: "0" },
        },
      },
      boxShadow: {
        "glow-sm": "0 0 12px rgba(59,130,246,0.15)",
        "glow-md": "0 0 24px rgba(59,130,246,0.2)",
        "glow-lg": "0 0 48px rgba(59,130,246,0.25)",
      },
      backgroundImage: {
        "grid-dark":
          "linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px)",
        "grid-light":
          "linear-gradient(rgba(29,78,216,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(29,78,216,0.04) 1px, transparent 1px)",
      },
      backgroundSize: {
        grid: "40px 40px",
      },
    },
  },
  plugins: [],
};
