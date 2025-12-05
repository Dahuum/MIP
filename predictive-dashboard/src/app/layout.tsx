import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "OCP Fan C07 - Predictive Maintenance System",
  description: "AI-powered predictive maintenance dashboard for OCP Fan C07. Real-time failure prediction using LSTM Deep Learning.",
  keywords: "OCP, predictive maintenance, machine learning, LSTM, fan monitoring, industrial AI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
