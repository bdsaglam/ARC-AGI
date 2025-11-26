from enum import Enum
import ast
import re
import json
from typing import List, Optional, Tuple

Grid = List[List[int]]

class GridFormat(str, Enum):
    STANDARD = "standard"
    SEMICOLON = "semicolon"
    XML = "xml"
    CSV = "csv"
    PYTHON = "python"
    SPARSE = "sparse"
    ASCII = "ascii"
    MASK = "mask"
    COMPACT = "compact"

ASCII_MAP = {0: '.', 1: '#', 2: 'x', 3: 'o', 4: '+', 5: '*', 6: '=', 7: '@', 8: '%', 9: '&'}
ASCII_REV = {v: k for k, v in ASCII_MAP.items()}

def format_grid(grid: Grid, fmt: GridFormat = GridFormat.STANDARD) -> str:
    if fmt == GridFormat.STANDARD:
        return "\n".join(" ".join(str(c) for c in row) for row in grid)
    elif fmt == GridFormat.SEMICOLON:
        return "; ".join(" ".join(str(c) for c in row) for row in grid)
    elif fmt == GridFormat.XML:
        return "\n".join(f"<row>{' '.join(str(c) for c in row)}</row>" for row in grid)
    elif fmt == GridFormat.CSV:
        return "\n".join(",".join(str(c) for c in row) for row in grid)
    elif fmt == GridFormat.PYTHON:
        return str(grid)
    elif fmt == GridFormat.SPARSE:
        coords = []
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if val != 0:
                    coords.append(f"({r},{c}):{val}")
        return ", ".join(coords)
    elif fmt == GridFormat.ASCII:
        return "\n".join("".join(ASCII_MAP.get(c, str(c)) for c in row) for row in grid)
    elif fmt == GridFormat.MASK:
        colors = sorted(list(set(c for row in grid for c in row)))
        parts = []
        for color in colors:
            if color == 0: continue
            mask = [[1 if c == color else 0 for c in row] for row in grid]
            mask_str = "\n".join(" ".join(str(x) for x in r) for r in mask)
            parts.append(f"Color {color}:\n{mask_str}")
        return "\n\n".join(parts)
    elif fmt == GridFormat.COMPACT:
        return "|".join("".join(str(c) for c in row) for row in grid)
    return str(grid)

def parse_grid_from_text(text: str, fmt: GridFormat = GridFormat.STANDARD) -> Grid:
    text = text.strip()
    
    if fmt == GridFormat.PYTHON:
        matches = re.findall(r"\[\[.*?\]\]", text, re.DOTALL)
        if matches:
            try:
                return ast.literal_eval(matches[-1])
            except:
                pass
        raise ValueError("Could not parse Python grid")

    if fmt == GridFormat.SPARSE:
        # Look for last block of coordinates
        # Heuristic: split by newlines, find lines with coords
        # Or just regex all coords and assume they belong to the result?
        # Risk: Input grid coords might be in prompt reflection.
        # We want the *last* output.
        # We can try to split text by "Output" or something, but that's model specific.
        # I'll assume the last continuous block of coords.
        all_coords = list(re.finditer(r"\(\d+,\d+\):\d+", text))
        if not all_coords:
             raise ValueError("No sparse coordinates found")
        
        # Group by proximity? Or just take the last N?
        # Hard to know.
        # I'll take all coords found in the LAST line that has coords?
        # Or all coords after the last "Output:" keyword?
        # I'll try to parse the *last* line that looks like a sparse list.
        lines = text.splitlines()
        for line in reversed(lines):
            if re.search(r"\(\d+,\d+\):\d+", line):
                matches = re.findall(r"\(\d+,\d+\):\d+", line)
                cells = []
                max_r, max_c = 0, 0
                for m in matches:
                    parts = m.split(":")
                    val = int(parts[1])
                    coord = parts[0][1:-1].split(",")
                    r, c = int(coord[0]), int(coord[1])
                    cells.append((r, c, val))
                    max_r = max(max_r, r)
                    max_c = max(max_c, c)
                grid = [[0] * (max_c + 1) for _ in range(max_r + 1)]
                for r, c, v in cells:
                    grid[r][c] = v
                return grid
        raise ValueError("Could not parse Sparse grid")

    if fmt == GridFormat.MASK:
        sections = re.split(r"Color (\d+):", text)
        if len(sections) < 3:
            # Fallback to standard parsing if mask structure is missing
            return parse_grid_from_text(text, fmt=GridFormat.STANDARD)

        masks = []
        for i in range(1, len(sections), 2):
            color = int(sections[i])
            content = sections[i + 1]
            mask_rows = []
            for line in content.strip().splitlines():
                tokens = line.split()
                if all(t.isdigit() for t in tokens):
                    mask_rows.append([int(t) for t in tokens])
            if mask_rows:
                masks.append((color, mask_rows))

        if not masks:
            raise ValueError("No valid masks found")

        rows = len(masks[0][1])
        cols = len(masks[0][1][0])
        result_grid = [[0] * cols for _ in range(rows)]

        for color, mask in masks:
            for r in range(min(rows, len(mask))):
                for c in range(min(cols, len(mask[r]))):
                    if mask[r][c] == 1:
                        result_grid[r][c] = color
        return result_grid

    if fmt == GridFormat.COMPACT:
        # Find last occurrence of `\d+\|\d+`
        candidates = re.findall(r"(?:\d+\|)+\d+", text)
        if candidates:
            block = candidates[-1]
            rows = block.split("|")
            return [[int(c) for c in row] for row in rows]
        raise ValueError("Could not parse Compact grid")

    # Line-based parsers (Standard, ASCII, CSV, Semicolon, XML)
    
    candidate_rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"): continue
        if not stripped: 
            candidate_rows.append(None)
            continue
            
        row = None
        if fmt == GridFormat.STANDARD:
            tokens = stripped.split()
            if all(t.isdigit() for t in tokens):
                row = [int(t) for t in tokens]
        elif fmt == GridFormat.CSV:
            tokens = stripped.split(",")
            if all(t.isdigit() for t in tokens):
                row = [int(t) for t in tokens]
        elif fmt == GridFormat.SEMICOLON:
            # Semicolon format is single line usually? "r1; r2; r3"
            # If text has newlines, we look for the line with semicolons
            if ";" in stripped:
                parts = stripped.split(";")
                try:
                    row_list = []
                    for p in parts:
                        sub_row = [int(t) for t in p.strip().split()]
                        row_list.append(sub_row)
                    # This parses a whole grid in one line!
                    # Return immediately if valid?
                    # But we want the *last* one.
                    # So we store it.
                    # Treat `row_list` as a "Block".
                    # How to fit into candidate_rows structure?
                    # candidate_rows expects single row.
                    # I'll handle SEMICOLON separately loop.
                    pass 
                except:
                    pass
        elif fmt == GridFormat.ASCII:
            # Check if chars are in map
            if all(c in ASCII_REV for c in stripped):
                row = [ASCII_REV[c] for c in stripped]
        elif fmt == GridFormat.XML:
            # <row>...</row>
            if stripped.startswith("<row>") and stripped.endswith("</row>"):
                content = stripped[5:-6]
                tokens = content.split()
                if all(t.isdigit() for t in tokens):
                    row = [int(t) for t in tokens]

        candidate_rows.append(row)

    if fmt == GridFormat.SEMICOLON:
        # Search for last valid semicolon line
        for line in reversed(text.splitlines()):
            if ";" in line:
                try:
                    parts = line.split(";")
                    grid = []
                    for p in parts:
                        grid.append([int(t) for t in p.strip().split()])
                    if grid: return grid
                except:
                    continue
        raise ValueError("Could not parse Semicolon grid")

    # Block reconstruction for row-based formats
    blocks = []
    current_block = []
    for row in candidate_rows:
        if row is None:
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            if current_block and len(row) != len(current_block[0]):
                blocks.append(current_block)
                current_block = [row]
            else:
                current_block.append(row)
    if current_block: blocks.append(current_block)
    
    if not blocks:
        raise ValueError(f"Could not parse {fmt} grid")
    
    return blocks[-1]

def verify_prediction(predicted: Grid, expected: Grid) -> bool:
    return predicted == expected