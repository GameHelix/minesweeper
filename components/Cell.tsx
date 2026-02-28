"use client";

import { CellState } from "@/lib/minesweeper";

interface CellProps {
  cell: CellState;
  row: number;
  col: number;
  isLosingCell: boolean;
  gameOver: boolean;
  onReveal: (row: number, col: number) => void;
  onFlag: (row: number, col: number) => void;
  onChord: (row: number, col: number) => void;
  onMouseDown: () => void;
  onMouseUp: () => void;
}

const NUMBER_COLORS: Record<number, string> = {
  1: "#1d4ed8",
  2: "#15803d",
  3: "#dc2626",
  4: "#1e3a8a",
  5: "#7f1d1d",
  6: "#0e7490",
  7: "#111827",
  8: "#6b7280",
};

export default function Cell({
  cell,
  row,
  col,
  isLosingCell,
  gameOver,
  onReveal,
  onFlag,
  onChord,
  onMouseDown,
  onMouseUp,
}: CellProps) {
  function handleClick() {
    if (gameOver) return;
    if (!cell.isRevealed && !cell.isFlagged) {
      onReveal(row, col);
    } else if (cell.isRevealed && cell.adjacentMines > 0) {
      onChord(row, col);
    }
  }

  function handleRightClick(e: React.MouseEvent) {
    e.preventDefault();
    if (gameOver || cell.isRevealed) return;
    onFlag(row, col);
  }

  // Pick CSS class
  let cellClass = "cell-unrev";
  if (cell.isRevealed) {
    cellClass = isLosingCell ? "cell-mine-loss" : "cell-rev";
  }

  // Content
  let content: React.ReactNode = null;
  if (cell.isFlagged && !cell.isRevealed) {
    content = <span className="select-none text-base leading-none">🚩</span>;
  } else if (cell.isRevealed) {
    if (cell.isMine) {
      content = <span className="select-none text-base leading-none">💣</span>;
    } else if (cell.adjacentMines > 0) {
      content = (
        <span
          className="font-bold select-none leading-none"
          style={{
            fontSize: "0.8rem",
            color: NUMBER_COLORS[cell.adjacentMines] ?? "#111",
          }}
        >
          {cell.adjacentMines}
        </span>
      );
    }
  }

  return (
    <button
      className={`w-8 h-8 flex items-center justify-center cursor-pointer transition-none ${cellClass}`}
      style={{ flexShrink: 0 }}
      onClick={handleClick}
      onContextMenu={handleRightClick}
      onMouseDown={gameOver ? undefined : onMouseDown}
      onMouseUp={gameOver ? undefined : onMouseUp}
      onMouseLeave={gameOver ? undefined : onMouseUp}
      disabled={gameOver && !cell.isRevealed}
    >
      {content}
    </button>
  );
}
