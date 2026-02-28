// ─── Types ────────────────────────────────────────────────────────────────────

export type CellState = {
  isMine: boolean;
  isRevealed: boolean;
  isFlagged: boolean;
  adjacentMines: number; // 0-8
};

export type Board = CellState[][];

export type GameStatus = "idle" | "playing" | "won" | "lost";

export type Difficulty = "beginner" | "intermediate" | "expert";

export type GameState = {
  board: Board;
  status: GameStatus;
  difficulty: Difficulty;
  flagCount: number; // flags placed
  elapsed: number; // seconds since first click
  startTime: number | null;
  losingCell: [number, number] | null; // cell that triggered loss
};

// ─── Constants ────────────────────────────────────────────────────────────────

export const DIFFICULTIES: Record<
  Difficulty,
  { rows: number; cols: number; mines: number }
> = {
  beginner: { rows: 9, cols: 9, mines: 10 },
  intermediate: { rows: 16, cols: 16, mines: 40 },
  expert: { rows: 16, cols: 30, mines: 99 },
};

// ─── Pure helpers ──────────────────────────────────────────────────────────────

function makeCell(): CellState {
  return { isMine: false, isRevealed: false, isFlagged: false, adjacentMines: 0 };
}

/** Returns all valid [r, c] neighbors of (row, col). */
function neighbors(
  board: Board,
  row: number,
  col: number
): [number, number][] {
  const rows = board.length;
  const cols = board[0].length;
  const result: [number, number][] = [];
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const r = row + dr;
      const c = col + dc;
      if (r >= 0 && r < rows && c >= 0 && c < cols) result.push([r, c]);
    }
  }
  return result;
}

// ─── Pure Functions ────────────────────────────────────────────────────────────

/** Create a blank board — no mines, nothing revealed. */
export function makeEmptyBoard(rows: number, cols: number): Board {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, makeCell)
  );
}

/**
 * Randomly place `mines` mines on the board, avoiding `safeRow`/`safeCol`
 * and all 8 of its neighbors.
 */
export function placeMines(
  board: Board,
  mines: number,
  safeRow: number,
  safeCol: number
): Board {
  const rows = board.length;
  const cols = board[0].length;

  // Build exclusion set
  const excluded = new Set<number>();
  excluded.add(safeRow * cols + safeCol);
  for (const [r, c] of neighbors(board, safeRow, safeCol)) {
    excluded.add(r * cols + c);
  }

  // Collect candidate positions
  const candidates: number[] = [];
  for (let i = 0; i < rows * cols; i++) {
    if (!excluded.has(i)) candidates.push(i);
  }

  // Fisher-Yates shuffle and take first `mines`
  for (let i = candidates.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
  }
  const mineSet = new Set(candidates.slice(0, mines));

  // Clone board and place mines
  return board.map((row, r) =>
    row.map((cell, c) => ({
      ...cell,
      isMine: mineSet.has(r * cols + c),
    }))
  );
}

/** Fill `adjacentMines` for every cell based on current mine positions. */
export function computeAdjacent(board: Board): Board {
  return board.map((row, r) =>
    row.map((cell, c) => {
      if (cell.isMine) return { ...cell, adjacentMines: 0 };
      const count = neighbors(board, r, c).filter(
        ([nr, nc]) => board[nr][nc].isMine
      ).length;
      return { ...cell, adjacentMines: count };
    })
  );
}

/**
 * Reveal cell (r, c). If it has 0 adjacent mines flood-fill BFS to reveal
 * all connected zero-adjacent and their numbered borders.
 * Does nothing if cell is already revealed or flagged.
 */
export function revealCell(board: Board, row: number, col: number): Board {
  const cell = board[row][col];
  if (cell.isRevealed || cell.isFlagged) return board;

  // Work on a mutable copy
  const next: Board = board.map((r) => r.map((c) => ({ ...c })));

  const queue: [number, number][] = [[row, col]];
  next[row][col].isRevealed = true;

  while (queue.length > 0) {
    const [r, c] = queue.shift()!;
    if (next[r][c].adjacentMines === 0 && !next[r][c].isMine) {
      for (const [nr, nc] of neighbors(next, r, c)) {
        if (!next[nr][nc].isRevealed && !next[nr][nc].isFlagged) {
          next[nr][nc].isRevealed = true;
          queue.push([nr, nc]);
        }
      }
    }
  }

  return next;
}

/** Toggle flag on an unrevealed cell. No-op if already revealed. */
export function toggleFlag(board: Board, row: number, col: number): Board {
  const cell = board[row][col];
  if (cell.isRevealed) return board;
  return board.map((r, ri) =>
    r.map((c, ci) =>
      ri === row && ci === col ? { ...c, isFlagged: !c.isFlagged } : c
    )
  );
}

/**
 * Chord reveal: if the number of flags around (row, col) equals adjacentMines,
 * reveal all un-flagged neighbors. Returns board unchanged if conditions not met.
 */
export function chordReveal(board: Board, row: number, col: number): Board {
  const cell = board[row][col];
  if (!cell.isRevealed || cell.adjacentMines === 0) return board;

  const nbrs = neighbors(board, row, col);
  const flagged = nbrs.filter(([r, c]) => board[r][c].isFlagged).length;
  if (flagged !== cell.adjacentMines) return board;

  let next = board;
  for (const [r, c] of nbrs) {
    if (!board[r][c].isFlagged && !board[r][c].isRevealed) {
      next = revealCell(next, r, c);
    }
  }
  return next;
}

/** Returns true when every non-mine cell has been revealed. */
export function checkWin(board: Board, totalMines: number): boolean {
  let revealed = 0;
  let total = 0;
  for (const row of board) {
    for (const cell of row) {
      total++;
      if (cell.isRevealed) revealed++;
    }
  }
  return revealed === total - totalMines;
}

/** Reveal all mines (called on loss). */
export function revealAllMines(board: Board): Board {
  return board.map((row) =>
    row.map((cell) =>
      cell.isMine ? { ...cell, isRevealed: true } : cell
    )
  );
}
