
# gpt-5.1-none is pretty much impossible to teach something. It largely doesn't respond to being supplied with a strategy.
I tested a few problems with very explicit solutions, and it didn't pick up on anything.

# gpt-5.1-low seems to respond to explicit strategies, but some are too complicated and hints-only is not working

## 1990f7a8
Strategy: NA
Pass rate: 1/10

Strategy:
```
Each input grid contains four separate 3×3 shapes made of color 2.
I crop each 3×3 shape (keeping its exact pattern), then place them into a new 7×7 grid as:
	•	top‑left: upper-left shape from the input
	•	top‑right: upper-right shape from the input
	•	bottom‑left: lower-left shape from the input
	•	bottom‑right: lower-right shape from the input
with a one‑cell row and column of zeros separating them (so shapes go in the 3×3 corners of the 7×7).
```
Pass rate: 4/10

Strategy: `Move an organize the objects. Create a separator between them.`\
Pass rate: 3/10

Strategy: `Move 3x3 objects to corners with an empty divider between them, creating a 7x7 output grid.`
Pass rate: 4/10

Strategy: `Hint 1: Move objects. Hint 2: Empty divider.`
Pass rate: 1/10

## 10fcaaa3

Strategy: NA
Pass rate: 0/10

Strategy:
```
First, tile the input grid into a 2×2 array of copies, doubling both height and width. Then, for every non-zero cell in this enlarged grid, keep its color, and place color 8 in each of its diagonal neighbors (up-left, up-right, down-left, down-right) as long as they stay within bounds and don’t overwrite a non-zero color.
```
Pass rate: 9/10

Strategy: `copy it, then expand into the corners`
Pass rate: 0/10

Strategy: `Copy four times. Expand into corners.`
Pass rate: 1/10

## 0a2355a6

Strategy: NA
Pass rate: 0/10

Strategy: `Each 8-blob is treated independently. For every 4-connected component of 8s, count how many “holes” (enclosed 0-regions fully surrounded by that component) it contains. Then recolor the entire component with a single color equal to that hole count: 1 → color 1, 2 → color 2, 3 → color 3, 4 → color 4. Background 0s stay 0.`
Pass rate: 0/10

Strategy:
```
1. Work one 8-region at a time
	•	Look at the grid and find each connected group of 8s.
	•	“Connected” means 4-connected: cells are neighbors if they touch up, down, left, or right (diagonals do not count).
	•	Each such group is a component that will be recolored to a single new color.

⸻

2. Define what a “hole” is

We only care about 0-cells (background) that are completely enclosed by one 8-component.

A hole is:
	•	A 4-connected region of 0s
	•	Such that every 4-neighbor of any cell in this 0-region is either:
	•	another 0 in the same region, or
	•	an 8 belonging to this same component

and no cell in the 0-region is connected (via 4-steps through 0s) to:
	•	any 0 on the outer border of the grid, or
	•	any 0 that touches another 8-component.

Intuition:
If you imagine coloring all the 8s in one component and treating everything else as “background,” a hole is a “lake” of 0s fully surrounded by that component — you cannot walk from that 0-region to the edge of the grid using only 0-steps.

⸻

3. How to count holes for one component (algorithmically)

For a given 8-component:
	1.	Mark all its 8-cells.
	2.	Consider the 0-cells that are adjacent to this component (or inside its shape).
	3.	Among those 0-cells, find all maximal 4-connected groups of 0s (using flood fill / BFS / DFS).
	4.	For each such 0-group:
	•	Temporarily ask: “If I start in this 0 and walk only on 0s, can I reach the grid boundary?”
	•	If yes, it is not a hole (it leaks to the outside).
	•	If no, it is a hole.
	5.	The hole count for that component is simply the number of these enclosed 0-groups.

⸻

4. Map “hole count” → new color

Once you have the hole count h for that 8-component:
	•	If h = 1, replace all 8s in that component with color 1.
	•	If h = 2, replace all 8s in that component with color 2.
	•	If h = 3, replace all 8s in that component with color 3.
	•	If h = 4, replace all 8s in that component with color 4.

(Those mappings come directly from the examples: components with 1, 2, 3, 4 holes consistently become colors 1, 2, 3, 4 respectively.)

All 0s stay 0.
```
Pass rate: 0/10


## 11dc524f

Strategy: NA
Pass rate: 0/10

Strategy: `Find the connected cluster of 2s and the cluster of 5s. Translate the 2-cluster in a straight line (horizontally or vertically) until its bounding box is directly adjacent to the 5-cluster’s bounding box. Then erase the original 5s and redraw them as a mirror image of the final 2-shape across the line between the two clusters, so the 2s and 5s form a symmetric pair.`
Pass rate: 9/10

Strategy: `Hint 1: Focus on connected clusters of same colors. Hint 2: Move things around. Hint 3: Create a mirror image.`
Pass rate: 0/10

# Test with successively less specific strategies

Strategy: `Identify the connected cluster of color 2 and the connected cluster of color 5. Determine their bounding boxes. Translate the 2-cluster in a straight line so its bounding box becomes directly adjacent to the 5-cluster. After positioning the 2s, erase the original 5s and redraw a new 5-shape that is the mirror image of the translated 2-shape across the boundary line between them. Leave all other cells as 7.`
Pass rate: 9/10

Strategy: `Locate two distinct non-background color clusters. Move the first cluster so that its bounding box touches the bounding box of the second cluster in either a horizontal or vertical direction. Then remove the second cluster’s original shape and recreate it as a mirrored copy of the moved first cluster, reflected across the line separating their bounding boxes.`
Pass rate: 6/10

Strategy: `When the grid contains two objects, reposition one so that it forms a deliberate adjacency with the other. After aligning them, replace the second object with a reflection of the first, ensuring the two shapes become symmetric partners. The background remains unchanged.`
Pass rate: 7/10

Strategy: `If the task contains two meaningful structures, consider repositioning or transforming one object, then generating a new object by applying a geometric transformation (like reflection or rotation) derived from the first. Many ARC tasks expect you to infer symmetry or alignment relationships between repeated or related shapes.`
Pass rate: 1/10

Strategy: `When faced with a problem that includes multiple elements, look for relationships between them. Try adjusting one element and then apply a consistent transformation to another so they form a coherent pattern. Use symmetry, alignment, or repetition to guide how elements should be repositioned or modified.`
Pass rate: 0/10


