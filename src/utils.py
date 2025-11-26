from typing import List

Grid = List[List[int]]

def format_grid(grid: Grid) -> str:
    """Formats a grid as CSV."""
    return "\n".join(",".join(str(c) for c in row) for row in grid)

def parse_grid_from_text(text: str) -> Grid:
    """Parses a CSV grid from text."""
    text = text.strip()
    
    candidate_rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"): continue
        if not stripped: 
            candidate_rows.append(None)
            continue
            
        row = None
        # CSV parsing
        tokens = stripped.split(",")
        # Allow for spaces around numbers just in case
        if all(t.strip().isdigit() for t in tokens):
            row = [int(t.strip()) for t in tokens]

        candidate_rows.append(row)

    # Block reconstruction
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
        raise ValueError("Could not parse grid")
    
    return blocks[-1]

def verify_prediction(predicted: Grid, expected: Grid) -> bool:
    return predicted == expected