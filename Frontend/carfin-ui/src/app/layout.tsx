import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CarFin AI - 당신만을 위한 AI 차량 추천",
  description: "AI 멀티에이전트가 제공하는 개인화 차량 추천 및 금융 상담 서비스",
  keywords: "차량추천, AI, 멀티에이전트, 금융상담, 중고차, 신차",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gradient-to-br from-slate-50 to-blue-50 min-h-screen`}
      >
        {children}
      </body>
    </html>
  );
}