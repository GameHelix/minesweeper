"use client";

import { Board as BoardType } from "@/lib/minesweeper";
import Cell from "./Cell";

interface BoardProps {
  board: BoardType;
  losingCell: [number, number] | null;
  gameOver: boolean;
  onReveal: (row: number, col: number) => void;
  onFlag: (row: number, col: number) => void;
  onChord: (row: number, col: number) => void;
  onCellMouseDown: () => void;
  onCellMouseUp: () => void;
}

export default function Board({
  board,
  losingCell,
  gameOver,
  onReveal,
  onFlag,
  onChord,
  onCellMouseDown,
  onCellMouseUp,
}: BoardProps) {
  return (
    /* overflow-x: auto lets the board scroll horizontally on small screens */
    <div className="overflow-x-auto" onContextMenu={(e) => e.preventDefault()}>
      <div className="inline-flex flex-col" style={{ gap: 1 }}>
        {board.map((row, r) => (
          <div key={r} className="flex" style={{ gap: 1 }}>
            {row.map((cell, c) => (
              <Cell
                key={c}
                cell={cell}
                row={r}
                col={c}
                isLosingCell={
                  losingCell !== null &&
                  losingCell[0] === r &&
                  losingCell[1] === c
                }
                gameOver={gameOver}
                onReveal={onReveal}
                onFlag={onFlag}
                onChord={onChord}
                onMouseDown={onCellMouseDown}
                onMouseUp={onCellMouseUp}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
