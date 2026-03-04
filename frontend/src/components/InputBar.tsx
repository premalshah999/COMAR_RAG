import { useEffect, useRef, useState } from "react";

interface Props {
  onSend: (text: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
}

function SendIcon() {
  return (
    <svg width={13} height={13} fill="none" stroke="currentColor" strokeWidth={2.5}
      strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
      <line x1={22} y1={2} x2={11} y2={13}/>
      <polygon points="22 2 15 22 11 13 2 9 22 2"/>
    </svg>
  );
}
function StopIcon() {
  return (
    <svg width={13} height={13} fill="currentColor" viewBox="0 0 24 24">
      <rect x={4} y={4} width={16} height={16} rx={2}/>
    </svg>
  );
}

export function InputBar({ onSend, onStop, isStreaming, disabled }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [value]);

  const handleSend = () => {
    const text = value.trim();
    if (!text || isStreaming) return;
    onSend(text);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const canSend = value.trim().length > 0 && !isStreaming && !disabled;

  return (
    <div className="input-area">
      <div className="input-wrap">
        <textarea
          ref={textareaRef}
          className="input-field"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Ask about Maryland regulations…"
          rows={1}
          disabled={disabled}
        />
        <div className="input-footer">
          <div className="input-footer-left" />
          {isStreaming ? (
            <button className="stop-btn" onClick={onStop} title="Stop generation">
              <StopIcon />
            </button>
          ) : (
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={!canSend}
              title="Send"
            >
              <SendIcon />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
