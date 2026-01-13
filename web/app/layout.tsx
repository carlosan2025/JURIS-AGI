import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "JURIS-AGI | VC Decision Intelligence",
  description: "Evidence-based VC investment decision analysis with counterfactual reasoning",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gray-50">
          <nav className="border-b bg-white">
            <div className="container mx-auto flex h-16 items-center px-4">
              <div className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
                  <span className="text-white font-bold text-sm">J</span>
                </div>
                <span className="font-semibold text-lg">JURIS-AGI</span>
                <span className="text-muted-foreground text-sm ml-2">VC Mode</span>
              </div>
              <div className="ml-auto flex items-center gap-4">
                <a href="/" className="text-sm text-muted-foreground hover:text-foreground">
                  Workspace
                </a>
                <a href="/extract" className="text-sm text-muted-foreground hover:text-foreground">
                  Extract
                </a>
                <a href="/context" className="text-sm text-muted-foreground hover:text-foreground">
                  Context
                </a>
                <a href="/reasoning" className="text-sm text-muted-foreground hover:text-foreground">
                  Reasoning
                </a>
                <a href="/audit" className="text-sm text-muted-foreground hover:text-foreground">
                  Audit
                </a>
              </div>
            </div>
          </nav>
          <main className="container mx-auto px-4 py-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
