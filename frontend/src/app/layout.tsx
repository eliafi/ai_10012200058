// Student: Eli Afi Ayekpley | Index: 10012200058
// CS4241 - Introduction to Artificial Intelligence | ACity 2026
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sankofa AI",
  description: "Retrieval-Augmented Generation for Ghana Elections & Budget Intelligence",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
