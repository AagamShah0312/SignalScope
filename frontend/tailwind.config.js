/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Near-black editorial canvas
        paper: "#0a0a0b",
        panel: "#101013",
        "panel-2": "#141417",
        line: "#1e1e22",
        "line-strong": "#2b2b31",
        // Text scale
        fg: "#ededef",
        muted: "#8b8b94",
        faint: "#565660",
        // Accent (lime) + verdict colors
        accent: "#bef264",
        ai: "#fbbf24",
        real: "#34d399",
      },
      fontFamily: {
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "SF Mono",
          "Menlo",
          "Consolas",
          "Liberation Mono",
          "monospace",
        ],
        serif: ["Georgia", "Cambria", "Times New Roman", "serif"],
      },
      letterSpacing: {
        label: "0.18em",
      },
    },
  },
  plugins: [],
};
