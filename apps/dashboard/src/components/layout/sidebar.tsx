"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  ArrowLeftRight,
  Brain,
  BarChart3,
  SlidersHorizontal,
  DollarSign,
  Bot,
  Menu,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/trades", label: "Trades", icon: ArrowLeftRight },
  { href: "/memory", label: "Memory", icon: Brain },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/params", label: "Parameters", icon: SlidersHorizontal },
  { href: "/costs", label: "Costs", icon: DollarSign },
];

export function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* Mobile hamburger button */}
      <button
        className="md:hidden fixed top-3 left-3 z-50 p-2 rounded-md bg-card border border-border"
        onClick={() => setOpen(!open)}
      >
        {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      {/* Overlay */}
      {open && (
        <div
          className="md:hidden fixed inset-0 bg-black/50 z-30"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "w-56 border-r border-border bg-card flex-col h-screen z-40",
          // Desktop: static in flex layout
          "hidden md:flex md:sticky md:top-0",
          // Mobile: fixed overlay, only visible when open
          open && "!flex fixed top-0 left-0 transition-transform duration-200"
        )}
      >
        <div className="p-4 border-b border-border">
          <Link href="/" className="flex items-center gap-2" onClick={() => setOpen(false)}>
            <Bot className="w-6 h-6 text-green-400" />
            <span className="font-bold text-lg">ScalpBot</span>
          </Link>
          <p className="text-xs text-muted-foreground mt-1">Adaptive Futures Scalper</p>
        </div>

        <nav className="flex-1 p-2 space-y-1">
          {links.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              onClick={() => setOpen(false)}
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
          Paper Trading Mode
        </div>
      </aside>
    </>
  );
}
