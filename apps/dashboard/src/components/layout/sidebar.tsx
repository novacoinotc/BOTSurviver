"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Bot,
  Inbox,
  BarChart3,
  GitBranch,
  Network,
} from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/requests", label: "Requests", icon: Inbox },
  { href: "/family-tree", label: "Family Tree", icon: GitBranch },
  { href: "/activity-map", label: "Mapa de Actividad", icon: Network },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-56 border-r border-border bg-card flex flex-col h-screen sticky top-0">
      <div className="p-4 border-b border-border">
        <Link href="/" className="flex items-center gap-2">
          <Bot className="w-6 h-6 text-green-400" />
          <span className="font-bold text-lg">BotSurviver</span>
        </Link>
      </div>

      <nav className="flex-1 p-2 space-y-1">
        {links.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
              pathname === href
                ? "bg-accent text-accent-foreground font-medium"
                : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
            )}
          >
            <Icon className="w-4 h-4" />
            {label}
          </Link>
        ))}
      </nav>

      <div className="p-4 border-t border-border text-xs text-muted-foreground">
        AI Survival Experiment
      </div>
    </aside>
  );
}
