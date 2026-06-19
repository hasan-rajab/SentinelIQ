import './globals.css';
import type { Metadata } from 'next';
import { Shield, Activity, Network, GitBranch } from 'lucide-react';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'SentinelIQ — Anomaly Intelligence',
  description: 'Multimodal AI anomaly detection for IT Ops & Cybersecurity',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-base-900 text-zinc-100 min-h-screen flex">
        {/* Nav rail */}
        <nav className="w-16 lg:w-56 border-r border-base-700 flex flex-col py-6 px-3 gap-1 shrink-0">
          <div className="flex items-center gap-2 px-2 mb-8">
            <Shield className="w-6 h-6 text-pulse shrink-0" strokeWidth={2.5} />
            <span className="hidden lg:block font-mono font-semibold text-sm tracking-tight">
              SENTINEL<span className="text-pulse">IQ</span>
            </span>
          </div>

          <NavLink href="/" icon={<Activity className="w-4 h-4" />} label="Live Feed" />
          <NavLink href="/alerts" icon={<Shield className="w-4 h-4" />} label="Alerts" />
          <NavLink href="/federated" icon={<Network className="w-4 h-4" />} label="Federated" />

          <div className="mt-auto px-2 hidden lg:block">
            <div className="flex items-center gap-2 text-xs text-zinc-500 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-pulse animate-pulse-slow" />
              v1.0.0
            </div>
          </div>
        </nav>

        <main className="flex-1 min-w-0">{children}</main>
      </body>
    </html>
  );
}

function NavLink({ href, icon, label }: { href: string; icon: React.ReactNode; label: string }) {
  return (
    <Link
      href={href}
      className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-base-800 transition-colors text-sm font-medium"
    >
      {icon}
      <span className="hidden lg:block">{label}</span>
    </Link>
  );
}
