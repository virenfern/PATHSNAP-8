import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PathSnap | Route Evaluation",
  description: "PathSnap GPS route matching evaluator interface",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
