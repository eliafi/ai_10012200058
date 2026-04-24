// Student: Eli Afi Ayekpley | Index: 10012200058
// CS4241 - Introduction to Artificial Intelligence | ACity 2026
import type { Metadata } from "next";
import { Inter, Playfair_Display } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const playfair = Playfair_Display({ subsets: ["latin"], weight: "700", variable: "--font-playfair" });

export const metadata: Metadata = {
  title: "Sankofa AI",
  description: "Retrieval-Augmented Generation for Ghana Elections & Budget Intelligence",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${playfair.variable}`}>
      <body style={{ fontFamily: "var(--font-inter), sans-serif" }}>{children}</body>
    </html>
  );
}
