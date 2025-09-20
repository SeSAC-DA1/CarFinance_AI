'use client';

export default function TestPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-blue-900 flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-white mb-4">
          🎯 CarFin AI Test Page
        </h1>
        <p className="text-gray-300">
          이 페이지가 정상 작동하면 기본 설정은 문제없습니다.
        </p>
        <div className="mt-8 p-4 bg-white/10 rounded-lg">
          <p className="text-green-400">✅ Next.js 15.5.3 실행 중</p>
          <p className="text-green-400">✅ Tailwind CSS 작동</p>
          <p className="text-green-400">✅ TypeScript 컴파일 성공</p>
        </div>
      </div>
    </div>
  );
}