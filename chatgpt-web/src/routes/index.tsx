import { createFileRoute } from "@tanstack/react-router";
import {
  ArrowUp,
  ChevronDown,
  CircleHelp,
  Images,
  Menu,
  Mic,
  MicOff,
  PanelLeft,
  PenLine,
  Plus,
  Search,
  Settings,
  Sparkle,
  Video,
  VideoOff,
  X,
} from "lucide-react";
import { useState } from "react";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  PromptInput,
  PromptInputButton,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import { Button } from "@/components/ui/button";
import { TooltipProvider } from "@/components/ui/tooltip";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "ChatGPT — AI Assistant" },
      { name: "description", content: "A visual recreation of the ChatGPT conversation workspace." },
      { property: "og:title", content: "ChatGPT — AI Assistant" },
      { property: "og:description", content: "A visual recreation of the ChatGPT conversation workspace." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: ChatWorkspace,
});

const navigation = [
  { label: "New chat", icon: PenLine, active: true },
  { label: "Search chats", icon: Search },
  { label: "Images", icon: Images },
];

function ChatWorkspace() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [voiceOpen, setVoiceOpen] = useState(false);

  return (
    <TooltipProvider>
    <main className="flex h-dvh overflow-hidden bg-background text-foreground">
      <aside
        className={`hidden shrink-0 border-r border-border bg-sidebar transition-[width] duration-200 md:flex md:flex-col ${sidebarOpen ? "w-[260px]" : "w-0 overflow-hidden border-r-0"}`}
      >
        <div className="flex h-14 items-center justify-between px-3">
          <div className="flex items-center gap-2 px-2 font-semibold">
            <OpenAILogo />
            <span>ChatGPT</span>
          </div>
          <Button aria-label="Close sidebar" className="size-9" onClick={() => setSidebarOpen(false)} size="icon" variant="ghost">
            <PanelLeft className="size-[18px]" />
          </Button>
        </div>

        <nav className="space-y-1 px-2" aria-label="Main navigation">
          {navigation.map(({ label, icon: Icon, active }) => (
            <Button
              className={`h-10 w-full justify-start gap-3 px-3 font-normal ${active ? "bg-sidebar-accent" : ""}`}
              key={label}
              variant="ghost"
            >
              <Icon className="size-[18px]" />
              {label}
            </Button>
          ))}
        </nav>

        <div className="mt-5 px-5 text-xs font-medium text-muted-foreground">Explore</div>
        <nav className="mt-2 space-y-1 px-2" aria-label="Explore">
          <Button className="h-10 w-full justify-start gap-3 px-3 font-normal" variant="ghost">
            <Sparkle className="size-[18px]" /> Explore GPTs
          </Button>
        </nav>

        <div className="mt-auto space-y-1 px-2 pb-3">
          <Button className="h-10 w-full justify-start gap-3 px-3 font-normal" variant="ghost">
            <CircleHelp className="size-[18px]" /> Help
          </Button>
          <Button className="h-10 w-full justify-start gap-3 px-3 font-normal" variant="ghost">
            <Settings className="size-[18px]" /> Settings
          </Button>
          <div className="mt-2 border-t border-border px-3 pt-4">
            <p className="text-sm font-medium">Get responses tailored to you</p>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">Log in to get answers based on saved chats, plus create images and upload files.</p>
            <Button className="mt-3 h-9 w-full rounded-full" variant="outline">Log in</Button>
          </div>
        </div>
      </aside>

      <section className="relative flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between px-3 md:px-4">
          <div className="flex items-center gap-1">
            {!sidebarOpen && (
              <Button aria-label="Open sidebar" className="mr-1 size-9" onClick={() => setSidebarOpen(true)} size="icon" variant="ghost">
                <PanelLeft className="size-[18px]" />
              </Button>
            )}
            <Button aria-label="Open menu" className="size-9 md:hidden" size="icon" variant="ghost">
              <Menu className="size-5" />
            </Button>
            <Button className="h-9 gap-1 px-2 text-base font-semibold" variant="ghost">
              ChatGPT <ChevronDown className="size-4 text-muted-foreground" />
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <Button className="h-9 rounded-full px-4">Log in</Button>
            <Button className="hidden h-9 rounded-full px-4 sm:inline-flex" variant="outline">Sign up for free</Button>
          </div>
        </header>

        <Conversation className="min-h-0">
          <ConversationContent className="h-full justify-center px-4 pb-32">
            <div className="mx-auto flex w-full max-w-3xl flex-col items-center">
              <h1 className="mb-8 text-center text-[28px] font-semibold leading-9">Where should we begin?</h1>
              <Composer onVoiceOpen={() => setVoiceOpen(true)} />
              <Button className="mt-5 h-9 rounded-full px-4" variant="outline">What can you do?</Button>
            </div>
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>

        <footer className="pointer-events-none absolute inset-x-0 bottom-2 px-4 text-center text-[11px] text-muted-foreground">
          ChatGPT can make mistakes. Check important info.
        </footer>
      </section>
      {voiceOpen && <VoiceMode onExit={() => setVoiceOpen(false)} />}
    </main>
    </TooltipProvider>
  );
}

function Composer({ onVoiceOpen }: { onVoiceOpen: () => void }) {
  return (
    <PromptInput
      className="w-full"
      onSubmit={({ text }) => {
        if (text.trim()) window.alert(`You asked: ${text}`);
      }}
    >
      <PromptInputTextarea className="min-h-[56px] py-4 pr-24 pl-12 text-base" placeholder="Ask anything" />
      <PromptInputFooter className="pointer-events-none absolute inset-0 flex h-full items-center px-2">
        <PromptInputTools className="pointer-events-auto">
          <PromptInputButton aria-label="Add files" className="size-9 rounded-full" tooltip="Add photos & files">
            <Plus className="size-5" />
          </PromptInputButton>
        </PromptInputTools>
        <div className="pointer-events-auto flex items-center gap-1">
          <PromptInputButton aria-label="Voice mode" className="size-9 rounded-full" onClick={onVoiceOpen} tooltip="Voice mode">
            <Mic className="size-5" />
          </PromptInputButton>
          <PromptInputSubmit className="size-9 rounded-full" status="ready">
            <ArrowUp className="size-5" />
          </PromptInputSubmit>
        </div>
      </PromptInputFooter>
    </PromptInput>
  );
}

type VoiceStatus = "Listening" | "Thinking" | "Speaking";

function VoiceMode({ onExit }: { onExit: () => void }) {
  const [muted, setMuted] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [status, setStatus] = useState<VoiceStatus>("Listening");

  const cycleStatus = () => {
    if (muted) return;
    setStatus((current) =>
      current === "Listening" ? "Thinking" : current === "Thinking" ? "Speaking" : "Listening",
    );
  };

  return (
    <div className="voice-overlay fixed inset-0 z-50 flex flex-col bg-voice text-voice-foreground animate-in fade-in duration-300">
      <header className="flex h-20 items-center justify-between px-5 sm:px-8">
        <div className="flex items-center gap-3 font-semibold">
          <OpenAILogo />
          <span>Voice</span>
        </div>
        <Button aria-label="Exit voice mode" className="size-11 rounded-full border-voice-border bg-voice-control text-voice-foreground hover:bg-voice-control-hover" onClick={onExit} size="icon" variant="outline">
          <X className="size-5" />
        </Button>
      </header>

      <div className="flex flex-1 flex-col items-center justify-center px-6 pb-20">
        {cameraOn && (
          <div className="absolute inset-x-5 top-24 h-36 overflow-hidden rounded-2xl border border-voice-border bg-voice-control sm:inset-x-auto sm:right-8 sm:h-40 sm:w-60">
            <div className="grid h-full place-items-center text-sm text-voice-muted">Camera preview</div>
          </div>
        )}

        <button
          aria-label={`Voice status: ${muted ? "Muted" : status}. Tap to change status.`}
          className={`voice-orb relative size-56 rounded-full sm:size-72 voice-${status.toLowerCase()} ${muted ? "voice-muted-orb" : ""}`}
          onClick={cycleStatus}
          type="button"
        >
          <span className="voice-orb-core absolute inset-[16%] rounded-full" />
          <span className="voice-orb-sheen absolute inset-[7%] rounded-full" />
        </button>

        <div className="mt-10 h-20 text-center">
          <h2 className="text-2xl font-medium">{muted ? "Muted" : status}</h2>
          <p className="mt-2 text-sm text-voice-muted">
            {muted ? "Tap the microphone to continue" : status === "Listening" ? "Go ahead, I’m listening" : status === "Thinking" ? "One moment…" : "ChatGPT is responding"}
          </p>
          {!muted && status !== "Thinking" && (
            <div className="voice-wave mt-5 flex h-6 items-center justify-center gap-1" aria-hidden="true">
              {[0, 1, 2, 3, 4, 5, 6].map((bar) => <span key={bar} />)}
            </div>
          )}
        </div>
      </div>

      <div className="absolute inset-x-0 bottom-8 flex items-center justify-center gap-5 sm:bottom-10">
        <div className="flex flex-col items-center gap-2">
          <Button
            aria-label={muted ? "Unmute microphone" : "Mute microphone"}
            className={`size-14 rounded-full border-voice-border text-voice-foreground ${muted ? "bg-voice-foreground text-voice" : "bg-voice-control hover:bg-voice-control-hover"}`}
            onClick={() => setMuted((value) => !value)}
            size="icon"
            variant="outline"
          >
            {muted ? <MicOff className="size-5" /> : <Mic className="size-5" />}
          </Button>
          <span className="text-xs text-voice-muted">{muted ? "Unmute" : "Mute"}</span>
        </div>
        <div className="flex flex-col items-center gap-2">
          <Button
            aria-label={cameraOn ? "Turn camera off" : "Turn camera on"}
            className={`size-14 rounded-full border-voice-border text-voice-foreground ${cameraOn ? "bg-voice-foreground text-voice" : "bg-voice-control hover:bg-voice-control-hover"}`}
            onClick={() => setCameraOn((value) => !value)}
            size="icon"
            variant="outline"
          >
            {cameraOn ? <VideoOff className="size-5" /> : <Video className="size-5" />}
          </Button>
          <span className="text-xs text-voice-muted">Camera</span>
        </div>
        <div className="flex flex-col items-center gap-2">
          <Button aria-label="Exit voice mode" className="size-14 rounded-full border-voice-danger bg-voice-danger text-voice-foreground hover:bg-voice-danger/90" onClick={onExit} size="icon" variant="outline">
            <X className="size-6" />
          </Button>
          <span className="text-xs text-voice-muted">End</span>
        </div>
      </div>
    </div>
  );
}

function OpenAILogo() {
  return (
    <div aria-hidden="true" className="grid size-6 place-items-center rounded-full border-2 border-foreground text-[9px] font-bold">
      AI
    </div>
  );
}