"use client"

import type React from "react"
import { useState, useRef, useEffect, useCallback, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { CodeiumEditor } from "@codeium/react-code-editor"
import { Send, Play, Save, Copy, RotateCcw, Code2, Sparkles, Zap, Heart } from "lucide-react"
import { useToast } from "@/components/ui/use-toast"
import { PHASE, isCompletedPhase } from "@/lib/phase"
import { useTheme } from "next-themes"
import ThemeToggle from "@/components/theme-toggle"
import { SettingsDialog } from "@/components/settings-dialog"
import { cn } from "@/lib/utils"
import "@/lib/i18n"

interface Message {
  bicep_code?: string
  content: string
  id: string
  sender: "user" | "assistant"
  timestamp: Date
}

interface ChatResponse {
  bicep_code?: string
  message: string
  phase: string
  requires_user_input: boolean
}

const INITIAL_CODE = `// ✨ Your Bicep template will appear here when generated from chat
// Welcome to your modern Bicep generator!` as const
const API_BASE_URL = "http://localhost:8000"
const CHAT_WIDTH_MIN = 320
const CHAT_WIDTH_MAX = 640
const CHAT_WIDTH_DEFAULT = 540

const generateSessionId = (): string => {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

const useResizable = (initialWidth: number, minWidth: number, maxWidth: number) => {
  const [width, setWidth] = useState(initialWidth)
  const [isResizing, setIsResizing] = useState(false)

  const handleMouseDown = useCallback(() => {
    setIsResizing(true)
  }, [])

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return
      const newWidth = Math.max(minWidth, Math.min(maxWidth, e.clientX))
      setWidth(newWidth)
    }

    const handleMouseUp = () => {
      setIsResizing(false)
    }

    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove)
      document.addEventListener("mouseup", handleMouseUp)
      document.body.style.cursor = "col-resize"
      document.body.style.userSelect = "none"
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }
  }, [isResizing, minWidth, maxWidth])

  return { width, isResizing, handleMouseDown }
}

const useChat = () => {
  const { t, i18n } = useTranslation()
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content: t("ui.greeting.initial_message"),
      sender: "assistant",
      timestamp: new Date(),
    },
  ])
  const [isLoading, setIsLoading] = useState(false)
  const [isSystemAdvancing, setIsSystemAdvancing] = useState(false)
  const [phase, setPhase] = useState<string>(PHASE.HEARING)
  const sessionIdRef = useRef<string>(generateSessionId())

  const sendMessageToAPI = useCallback(
    async (message: string): Promise<ChatResponse> => {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          message,
          language: i18n.language,
        }),
      })
      if (!response.ok) throw new Error(`API Error: ${response.status}`)
      return response.json()
    },
    [i18n.language],
  )

  const appendAssistantMessage = useCallback((response: ChatResponse) => {
    const aiMessage: Message = {
      id: `${Date.now()}-${Math.random()}`,
      content: response.message,
      sender: "assistant",
      timestamp: new Date(),
      bicep_code: response.bicep_code,
    }
    setMessages((prev) => [...prev, aiMessage])
  }, [])

  const resetConversation = useCallback(async () => {
    try {
      await fetch(`${API_BASE_URL}/reset`, { method: "POST" })
      sessionIdRef.current = generateSessionId()
      setMessages([
        {
          id: "1",
          content: t("ui.greeting.initial_message"),
          sender: "assistant",
          timestamp: new Date(),
        },
      ])
      setPhase(PHASE.HEARING)
      setIsSystemAdvancing(false)
      setIsLoading(false)
    } catch (error) {
      console.error("Failed to reset conversation:", error)
    }
  }, [t])

  return {
    messages,
    setMessages,
    isLoading,
    setIsLoading,
    isSystemAdvancing,
    setIsSystemAdvancing,
    phase,
    setPhase,
    sendMessageToAPI,
    appendAssistantMessage,
    resetConversation,
  }
}

export default function Home() {
  const { theme } = useTheme()
  const { toast } = useToast()
  const { t, i18n } = useTranslation()

  const {
    width: chatWidth,
    isResizing,
    handleMouseDown,
  } = useResizable(CHAT_WIDTH_DEFAULT, CHAT_WIDTH_MIN, CHAT_WIDTH_MAX)

  const {
    messages,
    setMessages,
    isLoading,
    setIsLoading,
    isSystemAdvancing,
    setIsSystemAdvancing,
    phase,
    setPhase,
    sendMessageToAPI,
    appendAssistantMessage,
    resetConversation,
  } = useChat()

  // Keep refs that always point to the latest state to avoid stale-closure issues
  const phaseRef = useRef(phase)
  useEffect(() => {
    phaseRef.current = phase
  }, [phase])

  const isSystemAdvancingRef = useRef(isSystemAdvancing)
  useEffect(() => {
    isSystemAdvancingRef.current = isSystemAdvancing
  }, [isSystemAdvancing])

  const isLoadingRef = useRef(isLoading)
  useEffect(() => {
    isLoadingRef.current = isLoading
  }, [isLoading])

  const [code, setCode] = useState<string>(INITIAL_CODE)
  const [inputMessage, setInputMessage] = useState("")
  const inputRef = useRef<HTMLInputElement | null>(null)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  const editorTheme = useMemo(() => (theme === "light" ? "light" : "vs-dark"), [theme])

  const phaseStatus = useMemo(() => {
    if (isSystemAdvancing)
      return {
        color: "bg-gradient-to-r from-primary to-secondary",
        animation: "animate-pulse animate-glow",
        icon: <Zap className="h-3 w-3 text-white" />,
        borderClass: "border-primary/30",
      }
    if (isCompletedPhase(phase))
      return {
        color: "bg-gradient-to-r from-green-400 to-emerald-500",
        animation: "animate-float",
        icon: <Heart className="h-3 w-3 text-white" />,
        borderClass: "border-green-400/30",
      }
    return {
      color: "bg-gradient-to-r from-primary to-amber-500",
      animation: "",
      icon: <Sparkles className="h-3 w-3 text-white" />,
      borderClass: "border-amber-500/30",
    }
  }, [isSystemAdvancing, phase])

  const handleLanguageChange = useCallback(
    (newLanguage: "ja" | "en") => {
      i18n.changeLanguage(newLanguage)
      setMessages((prev) => {
        const updatedMessages = [...prev]
        if (updatedMessages.length > 0 && updatedMessages[0].sender === "assistant") {
          updatedMessages[0] = {
            ...updatedMessages[0],
            content: t("ui.greeting.initial_message"),
          }
        }
        return updatedMessages
      })
    },
    [i18n, t, setMessages],
  )

  const processChatResponse = useCallback(
    (response: ChatResponse) => {
      const transitioningToCompleted = isCompletedPhase(response.phase) && !isCompletedPhase(phaseRef.current)

      appendAssistantMessage(response)
      if (response.phase) setPhase(response.phase)
      if (response.bicep_code) setCode(response.bicep_code)

      if (isCompletedPhase(response.phase) && !transitioningToCompleted) {
        setIsSystemAdvancing(false)
        return
      }

      if (response.requires_user_input) {
        setIsSystemAdvancing(false)
      } else {
        setIsSystemAdvancing(true)
        setTimeout(async () => {
          // if (isCompletedPhase(phaseRef.current)) return
          try {
            const resp = await sendMessageToAPI("")
            processChatResponse(resp)
          } catch (e) {
            console.error("Auto advance error", e)
            setIsSystemAdvancing(false)
          }
        }, 800)
      }
    },
    [appendAssistantMessage, setPhase, setCode, setIsSystemAdvancing, sendMessageToAPI],
  )

  const handleSendMessage = useCallback(async () => {
    if (!inputMessage.trim() || isLoading) return

    setIsSystemAdvancing(false)

    const newMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: "user",
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, newMessage])

    const userRequest = inputMessage
    setInputMessage("")
    setIsLoading(true)

    try {
      const response = await sendMessageToAPI(userRequest)
      processChatResponse(response)
    } catch (error) {
      console.error("Error sending message:", error)
      const errorResponse: Message = {
        id: `${Date.now()}-error`,
        content: `エラーが発生しました。バックエンドサーバーが起動していることを確認してください。`,
        sender: "assistant",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorResponse])
    } finally {
      setIsLoading(false)
    }
  }, [inputMessage, isLoading, setIsSystemAdvancing, setMessages, setIsLoading, sendMessageToAPI, processChatResponse])

  const handleKeyPress = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault()
        handleSendMessage()
      }
    },
    [handleSendMessage],
  )

  const copyToClipboard = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(code)
      toast({
        title: "✨ Copied to clipboard",
        description: "Your Bicep template is ready to use!",
      })
    } catch (error) {
      console.error("Failed to copy to clipboard:", error)
    }
  }, [code, toast])

  const downloadBicep = useCallback(() => {
    try {
      const blob = new Blob([code ?? ""], { type: "text/plain;charset=utf-8" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = "main.bicep"
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (e) {
      console.error("Failed to download bicep file", e)
    }
  }, [code])

  const handleRun = useCallback(() => {
    toast({
      title: "🚀 " + t("ui.toast.deploy_not_implemented"),
      description: t("ui.toast.deploy_description"),
      duration: 4000,
    })
  }, [toast, t])

  // Clear editor code when resetting the conversation
  const handleReset = useCallback(async () => {
    try {
      await resetConversation()
      setCode(INITIAL_CODE)
    } catch (error) {
      console.error("Error during reset:", error)
    }
  }, [resetConversation, setCode])

  // Auto focus when input becomes available
  useEffect(() => {
    if (!isLoading && !isSystemAdvancing && !isCompletedPhase(phase)) {
      const el = inputRef.current
      if (el && document.activeElement !== el) {
        requestAnimationFrame(() => el.focus())
      }
    }
  }, [isLoading, isSystemAdvancing, phase])

  // Refocus after send (input cleared) for quick consecutive messages
  useEffect(() => {
    if (inputMessage === "" && !isLoading && !isSystemAdvancing && !isCompletedPhase(phase)) {
      inputRef.current?.focus()
    }
  }, [inputMessage, isLoading, isSystemAdvancing, phase])

  // Auto-scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  return (
    <div className="h-screen bg-background text-foreground flex overflow-hidden">
      <div
        className="flex flex-col border-r border-border min-w-0 bg-card  mr-1"
        style={{ width: `${chatWidth}px` }}
      >
        <div className="h-16 bg-card border-b border-border flex items-center px-6 justify-between rounded-tr-2xl">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div className={cn("p-2 rounded-xl", phaseStatus.color, phaseStatus.animation)}>{phaseStatus.icon}</div>
              <div>
                <span className="text-sm font-semibold text-foreground">{t("ui.chat.assistant_name")}</span>
                <Badge variant="outline" className={cn("ml-2 text-xs", phaseStatus.borderClass)}>
                  {phase}
                </Badge>
              </div>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-xl"
            title={t("ui.chat.reset_conversation")}
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>

        <ScrollArea className="flex-1 p-6 h-0">
          <div className="space-y-3">
            {messages.map((message) => (
              <div key={message.id} className={cn("flex", message.sender === "user" ? "justify-end" : "justify-start")}>
                <div
                  className={cn(
                    "max-w-[90%] rounded-2xl px-5 py-3 text-sm whitespace-pre-wrap break-words shadow-sm",
                    message.sender === "user"
                      ? "bg-gradient-to-r from-secondary to-secondary/90 text-secondary-foreground ml-4 rounded-br-lg shadow-md"
                      : "bg-muted/50 text-foreground mr-4 rounded-bl-lg border border-border/50 shadow-md backdrop-blur-sm",
                  )}
                >
                  {message.content}
                </div>
              </div>
            ))}
            {(isLoading || isSystemAdvancing) && (
              <div className="flex justify-start">
                <div className="bg-muted/50 text-muted-foreground rounded-2xl rounded-bl-lg px-5 py-3 text-sm mr-4 border border-border/50 shadow-md backdrop-blur-sm">
                  <div className="flex items-center gap-3">
                    <div className="flex gap-1">
                      <div
                        className="w-2 h-2 bg-current rounded-full animate-bounce"
                        style={{ animationDelay: "0ms" }}
                      />
                      <div
                        className="w-2 h-2 bg-current rounded-full animate-bounce"
                        style={{ animationDelay: "150ms" }}
                      />
                      <div
                        className="w-2 h-2 bg-current rounded-full animate-bounce"
                        style={{ animationDelay: "300ms" }}
                      />
                    </div>
                    <span>{isSystemAdvancing ? t("ui.chat.auto_processing") : t("ui.chat.thinking")}</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        <Separator className="opacity-30" />

        <div className="p-6 bg-card border-t border-border rounded-br-2xl">
          <div className="flex gap-3">
            <Input
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={
                isCompletedPhase(phase) ? t("ui.chat.input_placeholder_completed") : t("ui.chat.input_placeholder")
              }
              className="flex-1 rounded-xl border-border bg-background focus:ring-2 focus:ring-secondary/20 focus:border-secondary/50"
              disabled={isLoading || isCompletedPhase(phase)}
            />
            <Button
              onClick={handleSendMessage}
              size="default"
              disabled={isLoading || isCompletedPhase(phase)}
              className="rounded-xl bg-gradient-to-r from-secondary to-secondary/90 hover:from-secondary/90 hover:to-secondary shadow-lg hover:shadow-xl transition-all duration-200"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
          {isCompletedPhase(phase) && (
            <p className="text-xs text-muted-foreground mt-3 text-center">{t("ui.chat.completion_notice")}</p>
          )}
        </div>
      </div>

      <div
        className={cn(
          "w-1 cursor-col-resize transition-all duration-200 hover:bg-gradient-to-b hover:from-secondary/30 hover:to-secondary/50 hidden md:block",
          isResizing && "bg-gradient-to-b from-secondary/50 to-secondary/70 w-2",
        )}
        onMouseDown={handleMouseDown}
      />

      <div className="flex-1 flex flex-col min-w-0 rounded-l-2xl ml-1 overflow-hidden">
        <div className="h-16 bg-yellow-50 dark:bg-card/30 border-b border-yellow-200 dark:border-border/50 flex items-center px-6 gap-4 rounded-tl-2xl border border-yellow-200 dark:border-border/50 border-b-0">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-r from-secondary/20 to-accent/20">
              <Code2 className="h-4 w-4 text-secondary" />
            </div>
            <span className="text-sm font-semibold">{t("ui.editor.filename")}</span>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={copyToClipboard}
              title={t("ui.editor.copy_tooltip")}
              className="text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-xl"
            >
              <Copy className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={downloadBicep}
              title={t("ui.editor.download_tooltip")}
              className="text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-xl"
            >
              <Save className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRun}
              title={t("ui.editor.deploy_tooltip")}
              className="text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-xl"
            >
              <Play className="h-4 w-4" />
            </Button>
            {/* text-muted-foreground と同じ色 */}
            <Separator orientation="vertical" />
            <ThemeToggle className="text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-xl" />
            <SettingsDialog onLanguageChange={handleLanguageChange} />
          </div>
        </div>

        <div className="flex-1 flex min-w-0 overflow-hidden bg-yellow-50/30 dark:bg-card/30 rounded-b-2xl border border-yellow-200 dark:border-border/50 border-t-0">
          <div className="flex-1 relative min-w-0 h-full overflow-hidden bg-yellow-50/20 dark:bg-card/20 rounded-b-2xl">
            {/* eslint-disable-next-line @typescript-eslint/ban-ts-comment */}
            {/* @ts-ignore */}
            <CodeiumEditor
              value={code}
              onChange={(value?: string) => setCode(value ?? "")}
              language="bicep"
              className="h-full w-full rounded-b-2xl"
              height="100%"
              theme={editorTheme}
              options={{
                bracketPairColorization: { enabled: true },
                fontSize: 15,
                tabSize: 2,
                // Force Consolas as primary editor font (fallbacks retained)
                fontFamily: "Consolas, 'JetBrains Mono', 'Courier New', monospace",
                lineNumbers: "on",
                minimap: { enabled: true },
                scrollBeyondLastLine: false,
                automaticLayout: true,
                wordWrap: "on",
                renderWhitespace: "selection",
                cursorBlinking: "smooth",
                cursorSmoothCaretAnimation: "on",
                smoothScrolling: true,
                padding: { top: 20, bottom: 20 },
              }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
