from typing import List, Optional

Grid = List[List[int]]

def format_grid(grid: Grid) -> str:
    """Formats a grid as CSV."""
    if grid is None:
        return ""
    return "\n".join(",".join(str(c) for c in row) for row in grid)

def parse_grid_from_text(text: str) -> Grid:
    """Parses a CSV grid from text, handling noise and labels.
    
    Strategies:
    1.  Identify all 'candidate rows' (lines containing only comma-separated numbers).
    2.  Group consecutive rows into 'blocks'.
    3.  Treat markdown code fences (```) as HARD separators that break blocks.
    4.  Allow small gaps (blank lines/text) within a block ONLY if no hard separators are encountered.
    5.  Return the LAST valid block found, assuming it is the final answer.
    """
    text = text.strip()
    lines = text.splitlines()
    
    candidate_rows = []
    hard_separators = [] # Track indices of lines that are hard separators
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Hard Separator Detection
        if stripped.startswith("```"): 
            candidate_rows.append(None)
            hard_separators.append(i)
            continue
            
        if not stripped: 
            candidate_rows.append(None)
            continue
            
        row = None
        # CSV parsing
        try:
            tokens = stripped.split(",")
            # Allow for spaces around numbers just in case
            if len(tokens) > 0 and all(t.strip().isdigit() for t in tokens):
                row = [int(t.strip()) for t in tokens]
        except ValueError:
            pass

        candidate_rows.append(row)

    # Block reconstruction
    blocks = []
    current_block = []
    
    # State tracking for the current block
    current_block_start_index = -1
    last_row_index = -1
    
    MAX_GAP = 3 # Allow a small gap of text/newlines within a grid (e.g. noise)
    
    for i, row in enumerate(candidate_rows):
        if row is not None:
            # We have a grid row
            if not current_block:
                # Start new block
                current_block = [row]
                current_block_start_index = i
                last_row_index = i
            else:
                # Existing block. Check compatibility.
                
                # 1. Check for Hard Separators between last row and this row
                has_hard_sep = any(last_row_index < sep_idx < i for sep_idx in hard_separators)
                
                # 2. Check for Gap Size
                gap_size = i - last_row_index - 1
                
                # 3. Check Width Compatibility
                width_match = len(row) == len(current_block[0])
                
                if has_hard_sep or gap_size > MAX_GAP or not width_match:
                    # Check for Exact Duplication before finalizing
                    # If the NEW block we are about to start is identical to the CURRENT block,
                    # we might be seeing a repetition.
                    # However, we can't see the future rows yet.
                    # Instead, let's look at the gap logic:
                    # If we are here, we are SPLITTING.
                    # So blocks.append(current_block) happens.
                    blocks.append(current_block)
                    current_block = [row]
                    current_block_start_index = i
                    last_row_index = i
                else:
                    # We are considering MERGING.
                    # Check if the new row is starting a repetition of the current block?
                    # This is hard to know line-by-line.
                    
                    # Alternative Strategy:
                    # If we merge, we might double the grid.
                    # Let's check if the 'current_block' so far consists of two identical halves?
                    # No, that's too expensive to check every row.
                    
                    # Let's rely on the block splitting logic we just added (Hard Separators).
                    # But here we have NO hard separator, just a blank line.
                    # And the content is identical. 
                    
                    # Refined Heuristic:
                    # If the gap is small (>0) AND the new row looks exactly like the START of the current block,
                    # AND the block is substantial size, it might be a repeat.
                    # But patterns repeat too!
                    
                    # Safest bet for 'Duplicate Block Detection':
                    # Post-process the blocks list? No, we need to decide whether to merge NOW.
                    
                    # Actually, if the user explicitly approved "Duplicate Block Detection", 
                    # we should probably split if there is ANY gap (gap_size > 0) 
                    # and allow the post-selection (taking the last block) to handle it.
                    # But that breaks "gappy" single grids.
                    
                    # Let's try this:
                    # If gap > 0, we tentatively verify if this new segment matches the start of the existing block.
                    # If row == current_block[0], it is HIGHLY suspicious of being a repeat start.
                    
                    is_repeat_start = (gap_size > 0) and (row == current_block[0])
                    
                    if is_repeat_start:
                         # Treat as a split
                        blocks.append(current_block)
                        current_block = [row]
                        current_block_start_index = i
                        last_row_index = i
                    else:
                        # Extend current block
                        current_block.append(row)
                        last_row_index = i
                    
    if current_block: blocks.append(current_block)
    
    if not blocks:
        raise ValueError("Could not parse grid")
    
    # Return the last block found
    return blocks[-1]

def verify_prediction(predicted: Grid, expected: Optional[Grid]) -> Optional[bool]:
    if expected is None:
        return None
    return predicted == expected