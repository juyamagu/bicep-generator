"use client"

import React, { useEffect, useState } from "react"
import { useTheme } from "next-themes"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { Sun, Moon } from "lucide-react"

export interface ThemeToggleProps {
  className?: string
}

export default function ThemeToggle({ className }: ThemeToggleProps) {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    // Prevent hydration mismatch: render a visually hidden placeholder until mounted
    return (
      <Button variant="ghost" size="sm" className={cn("invisible", className)}>
        <Sun className="h-4 w-4" />
      </Button>
    )
  }

  const current = theme ?? "dark"
  const isDark = current === "dark"

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className={cn("text-muted hover:text-app hover-bg-surface-2", className)}
      title={isDark ? "Switch to light theme" : "Switch to dark theme"}
    >
      {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  )
}
