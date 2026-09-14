/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0b1220",
        panel: "#111a2e",
        accent: "#38bdf8",
        "accent-dim": "#0ea5e9",
      },
    },
  },
  plugins: [],
};
