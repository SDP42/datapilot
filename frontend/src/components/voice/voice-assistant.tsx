"use client";

import { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Mic, MicOff, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface SpeechRecognitionResultLike {
  isFinal: boolean;
  [index: number]: { transcript: string };
}

interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResultLike>;
}

interface SpeechRecognitionLike extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
}

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  }
}

interface Command {
  phrases: string[];
  description: string;
  run: (router: ReturnType<typeof useRouter>) => string; // returns the spoken confirmation
}

const COMMANDS: Command[] = [
  {
    phrases: ["overview", "home", "dashboard home"],
    description: "\"go to overview\"",
    run: (r) => {
      r.push("/dashboard");
      return "Opening the overview.";
    },
  },
  {
    phrases: ["all in one", "run everything", "full pipeline"],
    description: "\"run everything\"",
    run: (r) => {
      r.push("/dashboard/all-in-one");
      return "Opening the all-in-one pipeline.";
    },
  },
  {
    phrases: ["ingest", "upload"],
    description: "\"open ingest\"",
    run: (r) => {
      r.push("/dashboard/upload");
      return "Opening dataset ingestion.";
    },
  },
  {
    phrases: ["quality"],
    description: "\"open quality\"",
    run: (r) => {
      r.push("/dashboard/quality");
      return "Opening quality analysis.";
    },
  },
  {
    phrases: ["eda", "exploratory"],
    description: "\"open EDA\"",
    run: (r) => {
      r.push("/dashboard/eda");
      return "Opening exploratory data analysis.";
    },
  },
  {
    phrases: ["dashboards", "dashboard builder", "bi dashboard"],
    description: "\"open dashboards\"",
    run: (r) => {
      r.push("/dashboard/dashboards");
      return "Opening the dashboard builder.";
    },
  },
  {
    phrases: ["modeling", "model search", "train model"],
    description: "\"open modeling\"",
    run: (r) => {
      r.push("/dashboard/modeling");
      return "Opening modeling.";
    },
  },
  {
    phrases: ["predict", "prediction"],
    description: "\"open predict\"",
    run: (r) => {
      r.push("/dashboard/predict");
      return "Opening train and predict.";
    },
  },
  {
    phrases: ["jobs"],
    description: "\"open jobs\"",
    run: (r) => {
      r.push("/dashboard/jobs");
      return "Opening background jobs.";
    },
  },
  {
    phrases: ["analytics"],
    description: "\"open analytics\"",
    run: (r) => {
      r.push("/dashboard/analytics");
      return "Opening analytics.";
    },
  },
  {
    phrases: ["history"],
    description: "\"open history\"",
    run: (r) => {
      r.push("/dashboard/history");
      return "Opening run history.";
    },
  },
];

function matchCommand(transcript: string): Command | null {
  const lower = transcript.toLowerCase();
  for (const cmd of COMMANDS) {
    if (cmd.phrases.some((p) => lower.includes(p))) return cmd;
  }
  return null;
}

function speak(text: string) {
  if (typeof window === "undefined" || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.05;
  window.speechSynthesis.speak(utterance);
}

export function VoiceAssistant() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [feedback, setFeedback] = useState<string | null>(null);
  const [supported, setSupported] = useState(
    () => typeof window !== "undefined" && Boolean(window.SpeechRecognition ?? window.webkitSpeechRecognition),
  );
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);

  const handleResult = useCallback(
    (text: string) => {
      setTranscript(text);
      const command = matchCommand(text);
      if (command) {
        const confirmation = command.run(router);
        setFeedback(confirmation);
        speak(confirmation);
      } else {
        const message = "Sorry, I didn't catch a known command.";
        setFeedback(message);
        speak(message);
      }
    },
    [router],
  );

  function startListening() {
    const Ctor = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!Ctor) {
      setSupported(false);
      return;
    }
    const recognition = new Ctor();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      const result = event.results[event.resultIndex];
      const text = result?.[0]?.transcript ?? "";
      handleResult(text);
    };
    recognition.onerror = () => {
      setListening(false);
      setFeedback("Microphone error — check browser permissions.");
    };
    recognition.onend = () => setListening(false);

    recognitionRef.current = recognition;
    setTranscript("");
    setFeedback(null);
    setListening(true);
    recognition.start();
  }

  function stopListening() {
    recognitionRef.current?.stop();
    setListening(false);
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={cn(
          "fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center rounded-full",
          "bg-gradient-to-br from-primary to-primary-2 text-white shadow-xl shadow-primary/40",
          "transition-transform hover:scale-105 active:scale-95",
        )}
        aria-label="Voice assistant"
      >
        {listening ? <Mic className="h-6 w-6 animate-pulse" /> : <Mic className="h-6 w-6" />}
      </button>

      {open && (
        <div className="fixed bottom-24 right-6 z-50 w-80 rounded-2xl border border-surface-border bg-surface/95 p-5 shadow-2xl backdrop-blur-xl">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold">Voice assistant</h3>
            <button onClick={() => setOpen(false)} className="text-muted hover:text-foreground">
              <X className="h-4 w-4" />
            </button>
          </div>

          {!supported ? (
            <p className="text-xs text-muted">
              Speech recognition isn&apos;t available in this browser. Try Chrome on desktop or
              Android.
            </p>
          ) : (
            <>
              <button
                type="button"
                onClick={listening ? stopListening : startListening}
                className={cn(
                  "flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-medium transition-colors",
                  listening
                    ? "bg-danger/15 text-danger"
                    : "bg-gradient-to-r from-primary to-primary-2 text-white",
                )}
              >
                {listening ? (
                  <>
                    <MicOff className="h-4 w-4" /> Stop listening
                  </>
                ) : (
                  <>
                    <Mic className="h-4 w-4" /> Tap to speak
                  </>
                )}
              </button>

              {transcript && (
                <p className="mt-3 rounded-lg bg-surface-2/60 px-3 py-2 text-xs text-muted">
                  &ldquo;{transcript}&rdquo;
                </p>
              )}
              {feedback && <p className="mt-2 text-xs text-primary-2">{feedback}</p>}

              <div className="mt-4 border-t border-surface-border pt-3">
                <p className="mb-1.5 text-xs font-medium text-muted">Try saying:</p>
                <div className="flex flex-wrap gap-1.5">
                  {COMMANDS.slice(0, 6).map((c) => (
                    <span
                      key={c.description}
                      className="rounded-full bg-surface-2/60 px-2 py-0.5 text-[10px] text-muted"
                    >
                      {c.description}
                    </span>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
}
