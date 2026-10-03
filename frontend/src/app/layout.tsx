import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { ThemeProvider } from "@/components/theme/theme-provider";
import { ThemedToaster } from "@/components/theme/themed-toaster";
import "./globals.css";

const THEME_INIT_SCRIPT = `
try {
  var theme = localStorage.getItem("dp_theme");
  document.documentElement.dataset.theme = theme === "light" ? "light" : "dark";
} catch (e) {
  document.documentElement.dataset.theme = "dark";
}
`;

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DataPilot — Autonomous Data Science Platform",
  description:
    "Deterministic ingestion, quality, EDA, modeling, and an autonomous agent — one platform, fully traced.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body className="min-h-full flex flex-col bg-background text-foreground">
        <ThemeProvider>
          {children}
          <ThemedToaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
