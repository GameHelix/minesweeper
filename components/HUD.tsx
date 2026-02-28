"use client";

import { GameStatus } from "@/lib/minesweeper";

interface HUDProps {
  minesRemaining: number;
  elapsed: number;
  status: GameStatus;
  isMouseDown: boolean;
  onNewGame: () => void;
}

function pad3(n: number): string {
  return String(Math.min(999, Math.max(0, n))).padStart(3, "0");
}

function formatTime(seconds: number): string {
  const s = Math.min(999, seconds);
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

function getFace(status: GameStatus, isMouseDown: boolean): string {
  if (status === "won") return "😎";
  if (status === "lost") return "💀";
  if (isMouseDown) return "😮";
  return "😊";
}

export default function HUD({
  minesRemaining,
  elapsed,
  status,
  isMouseDown,
  onNewGame,
}: HUDProps) {
  return (
    <div
      className="flex justify-between items-center mb-3 px-3 py-2 rounded-lg"
      style={{
        background: "#8a96a8",
        boxShadow:
          "inset -3px -3px 0 #5a636f, inset 3px 3px 0 #c0cad8",
      }}
    >
      {/* Mine counter */}
      <div className="led-display">{pad3(minesRemaining)}</div>

      {/* Face button */}
      <button
        onClick={onNewGame}
        title="New Game"
        className="text-2xl select-none rounded transition-transform active:scale-90"
        style={{
          width: "2.2rem",
          height: "2.2rem",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#a8b4c4",
          boxShadow:
            "inset -2px -2px 0 #6b7a8d, inset 2px 2px 0 #dde4ef",
          cursor: "pointer",
          fontSize: "1.25rem",
          lineHeight: 1,
        }}
      >
        {getFace(status, isMouseDown)}
      </button>

      {/* Timer */}
      <div className="led-display">{formatTime(elapsed)}</div>
    </div>
  );
}
