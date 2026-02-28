"use client";

import { useReducer, useEffect, useRef, useState, useCallback } from "react";
import {
  GameState,
  Difficulty,
  DIFFICULTIES,
  makeEmptyBoard,
  placeMines,
  computeAdjacent,
  revealCell,
  toggleFlag,
  chordReveal,
  checkWin,
  revealAllMines,
} from "@/lib/minesweeper";
import HUD from "./HUD";
import Board from "./Board";

// ─── Reducer ──────────────────────────────────────────────────────────────────

type Action =
  | { type: "REVEAL"; row: number; col: number }
  | { type: "FLAG"; row: number; col: number }
  | { type: "CHORD"; row: number; col: number }
  | { type: "NEW_GAME" }
  | { type: "SET_DIFFICULTY"; difficulty: Difficulty }
  | { type: "TICK" };

function makeInitialState(difficulty: Difficulty): GameState {
  const { rows, cols } = DIFFICULTIES[difficulty];
  return {
    board: makeEmptyBoard(rows, cols),
    status: "idle",
    difficulty,
    flagCount: 0,
    elapsed: 0,
    startTime: null,
    losingCell: null,
  };
}

function reducer(state: GameState, action: Action): GameState {
  switch (action.type) {
    case "NEW_GAME":
      return makeInitialState(state.difficulty);

    case "SET_DIFFICULTY":
      return makeInitialState(action.difficulty);

    case "TICK":
      if (state.status !== "playing") return state;
      return { ...state, elapsed: state.elapsed + 1 };

    case "FLAG": {
      if (state.status === "won" || state.status === "lost") return state;
      const { row, col } = action;
      const cell = state.board[row][col];
      if (cell.isRevealed) return state;

      const wasIdle = state.status === "idle";
      const newBoard = toggleFlag(state.board, row, col);
      const delta = newBoard[row][col].isFlagged ? 1 : -1;
      return {
        ...state,
        board: newBoard,
        flagCount: state.flagCount + delta,
        status: wasIdle ? "playing" : state.status,
      };
    }

    case "REVEAL": {
      if (state.status === "won" || state.status === "lost") return state;
      const { row, col } = action;
      const cell = state.board[row][col];
      if (cell.isRevealed || cell.isFlagged) return state;

      let board = state.board;
      let status = state.status;

      // First click — place mines and compute adjacency
      if (status === "idle") {
        const { mines } = DIFFICULTIES[state.difficulty];
        board = placeMines(board, mines, row, col);
        board = computeAdjacent(board);
        status = "playing";
      }

      board = revealCell(board, row, col);

      // Check loss: revealed a mine
      if (board[row][col].isMine) {
        return {
          ...state,
          board: revealAllMines(board),
          status: "lost",
          losingCell: [row, col],
        };
      }

      // Check win
      const { mines } = DIFFICULTIES[state.difficulty];
      if (checkWin(board, mines)) {
        return { ...state, board, status: "won" };
      }

      return { ...state, board, status };
    }

    case "CHORD": {
      if (state.status !== "playing") return state;
      const { row, col } = action;
      const board = chordReveal(state.board, row, col);

      // Check if any mine was revealed during chord
      for (let r = 0; r < board.length; r++) {
        for (let c = 0; c < board[r].length; c++) {
          if (board[r][c].isMine && board[r][c].isRevealed) {
            return {
              ...state,
              board: revealAllMines(board),
              status: "lost",
              losingCell: [r, c],
            };
          }
        }
      }

      const { mines } = DIFFICULTIES[state.difficulty];
      if (checkWin(board, mines)) {
        return { ...state, board, status: "won" };
      }

      return { ...state, board };
    }

    default:
      return state;
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Game() {
  const [state, dispatch] = useReducer(reducer, makeInitialState("beginner"));
  const [isMouseDown, setIsMouseDown] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Timer
  useEffect(() => {
    if (state.status === "playing") {
      intervalRef.current = setInterval(() => dispatch({ type: "TICK" }), 1000);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [state.status]);

  const handleReveal = useCallback((row: number, col: number) => {
    dispatch({ type: "REVEAL", row, col });
  }, []);

  const handleFlag = useCallback((row: number, col: number) => {
    dispatch({ type: "FLAG", row, col });
  }, []);

  const handleChord = useCallback((row: number, col: number) => {
    dispatch({ type: "CHORD", row, col });
  }, []);

  const handleNewGame = useCallback(() => {
    dispatch({ type: "NEW_GAME" });
  }, []);

  const handleSetDifficulty = useCallback((difficulty: Difficulty) => {
    dispatch({ type: "SET_DIFFICULTY", difficulty });
  }, []);

  const { mines } = DIFFICULTIES[state.difficulty];
  const minesRemaining = mines - state.flagCount;

  const DIFFICULTY_LABELS: Record<Difficulty, string> = {
    beginner: "Beginner",
    intermediate: "Intermediate",
    expert: "Expert",
  };

  return (
    <div className="flex flex-col items-center gap-6 py-8">
      <h1 className="text-3xl font-bold tracking-tight text-slate-800">
        Minesweeper
      </h1>

      {/* Difficulty selector */}
      <div className="flex gap-2">
        {(["beginner", "intermediate", "expert"] as Difficulty[]).map((d) => (
          <button
            key={d}
            onClick={() => handleSetDifficulty(d)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              state.difficulty === d
                ? "bg-slate-700 text-white"
                : "bg-slate-200 text-slate-700 hover:bg-slate-300"
            }`}
          >
            {DIFFICULTY_LABELS[d]}
          </button>
        ))}
      </div>

      {/* Game panel */}
      <div className="bg-slate-300 p-4 rounded-xl shadow-md">
        <HUD
          minesRemaining={minesRemaining}
          elapsed={state.elapsed}
          status={state.status}
          isMouseDown={isMouseDown}
          onNewGame={handleNewGame}
        />
        <Board
          board={state.board}
          losingCell={state.losingCell}
          gameOver={state.status === "won" || state.status === "lost"}
          onReveal={handleReveal}
          onFlag={handleFlag}
          onChord={handleChord}
          onCellMouseDown={() => setIsMouseDown(true)}
          onCellMouseUp={() => setIsMouseDown(false)}
        />
      </div>

      {/* Status banner */}
      {state.status === "won" && (
        <div className="text-green-700 font-semibold text-lg animate-pulse">
          You won! 🎉
        </div>
      )}
      {state.status === "lost" && (
        <div className="text-red-700 font-semibold text-lg">
          Game over! 💀
        </div>
      )}
    </div>
  );
}
