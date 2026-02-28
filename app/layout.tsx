import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Minesweeper",
  description: "Classic Minesweeper built with Next.js",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={`${inter.className} min-h-screen antialiased`}
        style={{
          background: "radial-gradient(ellipse at top, #1e3a5f 0%, #0f172a 60%)",
          minHeight: "100dvh",
        }}
      >
        {children}
      </body>
    </html>
  );
}
