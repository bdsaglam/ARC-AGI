# Documentation: Generating the Cartoon-Style ARC Visualization

This document provides a comprehensive, step-by-step explanation of how the `_cartoon.png` image file is generated from an ARC (Abstract Reasoning Corpus) task JSON file. This specific image visualizes all the "train" input/output pairs from the JSON file in a vertically-stacked, "cartoonish" or hand-drawn style.

## 1. Core Dependencies

The script relies on two primary Python libraries for data manipulation and plotting:

-   **NumPy:** Used to represent the input and output grids as efficient, multi-dimensional arrays.
-   **Matplotlib:** The core plotting library used for all visualizations. We make use of its `pyplot` interface, `colors` and `patches` for drawing, and the advanced `GridSpec` for layout management.

```python
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import json
import os
import matplotlib.patheffects as path_effects
```

## 2. File Parsing

The process begins by reading the input JSON file (e.g., `task.json`). The script specifically looks for the `"train"` key, which is expected to contain a list of task objects. Each object in this list must have an `"input"` and an `"output"` key, which correspond to the 2D arrays representing the grids.

The `parse_json_task` function handles this. It opens the JSON file, deserializes it, and iterates through the `"train"` list, converting each grid into a NumPy array. The final result is a list of tuples, where each tuple contains an `(input_grid, output_grid)` pair.

```python
def parse_json_task(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    train_tasks = []
    for pair in data.get('train', []):
        train_tasks.append((np.array(pair['input']), np.array(pair['output'])))
    # (The function also parses 'test' tasks, but they are not used for this specific image)
    return train_tasks, [] 
```

## 3. Dynamic Figure Sizing

To ensure the final image has minimal whitespace and is sized appropriately for the content, the script **measures** the required dimensions before creating the plot.

1.  **Cell Size:** We define a constant pixel size for each cell in a grid (e.g., `CELL_PIXELS = 15`).
2.  **Calculate Total Content Height:** The code iterates through all training pairs and finds the maximum grid height (number of rows) for each pair. These heights are summed up to get the total number of vertical cells required.
3.  **Calculate Maximum Content Width:** It also finds the "busiest" row by summing the widths (number of columns) of the input and output grids for each pair and taking the maximum.
4.  **Add Padding & Convert to Inches:** A global padding factor (`PADDING_FACTOR = 1.2`) is applied to the total pixel height and width to leave a small margin for titles and spacing. This final pixel dimension is then divided by the desired Dots Per Inch (`DPI = 100`) to get the `figsize` in inches, which is what Matplotlib requires.

```python
# From the `plot_multiple_tasks_cartoon` function:
height_ratios = [max(i.shape[0], o.shape[0]) for i, o in tasks]
total_height_cells = sum(height_ratios)
max_width_cells = max([i.shape[1] + o.shape[1] for i, o in tasks])

# Determine figure size in inches based on content
fig_height_px = total_height_cells * CELL_PIXELS * PADDING_FACTOR
fig_width_px = max_width_cells * CELL_PIXELS * PADDING_FACTOR
fig = plt.figure(figsize=(fig_width_px / DPI, fig_height_px / DPI))
```

## 4. Advanced Layout with `GridSpec`

Instead of a simple subplot grid, we use Matplotlib's `GridSpec` for a more robust layout. This is crucial for handling grids of different sizes in the same image.

-   A `GridSpec` is created with 2 columns and a number of rows equal to the number of training pairs.
-   The `height_ratios` calculated in the previous step are passed to the `GridSpec`. This tells Matplotlib to allocate vertical space for each row proportionally to the height of the grids in that row, which is key to minimizing whitespace.
-   A horizontal spacing parameter (`hspace`) is used to control the vertical gap between the plots.

```python
# gs is a GridSpec object that defines the layout of the figure
gs = fig.add_gridspec(num_pairs, 2, height_ratios=height_ratios, hspace=0.5)
```

## 5. The Cartoon Style

The entire plotting process is wrapped in a `with plt.xkcd()` block. This context manager automatically applies a number of stylistic changes to all plotting commands within it:
-   Lines become "wobbly" and hand-drawn.
-   Fonts are changed to a handwritten style (if available on the system).
-   **Path Effects** are applied to lines, which is the key to the border style.

### Drawing a Single Grid (`_draw_cartoon_grid`)

This is the core helper function responsible for drawing a single input or output grid.

#### 5.1. The Background and Cells
First, it uses `ax.imshow()` to draw the colored cells of the grid. `interpolation='nearest'` ensures each cell is a sharp, colored square, and `aspect='equal'` forces the cells to be square.

```python
ax.imshow(grid, cmap=CMAP, norm=NORM, interpolation='nearest', zorder=0, aspect='equal')
```

#### 5.2. The Separator Lines (The "White Area with a Black Line")

The most complex part of the style is the separator line. The effect of a thin black line inside a slightly thicker "white area" is achieved by overriding the default `xkcd` path effect.

The `xkcd` style automatically draws a thick white line underneath any plotted line. To gain control over this, we define our own custom effect:

1.  We define a `PathEffect` that consists of a white stroke (`withStroke`) with a `linewidth` of **1.5**.
2.  We then draw our main black border line with a `linewidth` of **1**.

When rendered, Matplotlib first draws the 1.5pt white line, and then draws the 1pt black line on top of it. This leaves a small, 0.25pt white border visible on either side of the black line, creating the desired effect.

This effect is only applied where the color of adjacent cells changes, or at the outer boundary of the grid.

```python
# From the `_draw_cartoon_grid` function:

# Define the custom path effect for the borders
border_width = 1
border_color = 'black'
effects = [path_effects.withStroke(linewidth=1.5, foreground='w')]

# Loop through grid cells to find where colors change
for r in range(rows + 1):
    for c in range(cols):
        if r == 0 or r == rows or (r < rows and grid[r-1, c] != grid[r, c]):
            # Apply the custom effect when drawing the line
            ax.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], 
                    color=border_color, lw=border_width, zorder=10, path_effects=effects)
```

## 6. Assembling the Final Image

The main function, `plot_multiple_tasks_cartoon`, iterates through each training pair and performs the following for each:
1.  Creates a pair of subplots (`ax_in`, `ax_out`) for the row using the `GridSpec`.
2.  Calls `_draw_cartoon_grid` on each subplot to draw the input and output grids.
3.  Draws the connecting arrow between the two subplots.
4.  Finally, it calls `gs.tight_layout(fig)` to neatly adjust the spacing and saves the complete figure to the `{filename}_cartoon.png` file, using the specified DPI to ensure the dimensions are correct.

```python
# The main loop
for i, (input_grid, output_grid) in enumerate(tasks):
    ax_in = fig.add_subplot(gs[i, 0])
    _draw_cartoon_grid(ax_in, input_grid, f"Input {i+1}")
    ax_out = fig.add_subplot(gs[i, 1])
    _draw_cartoon_grid(ax_out, output_grid, f"Output {i+1}")
    # ... (arrow drawing logic) ...

# Final save
gs.tight_layout(fig, pad=1.0)
plt.savefig(output_filename, dpi=DPI)
```
