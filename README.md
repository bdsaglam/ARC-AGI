# How to get started

```bash
git clone https://github.com/beetree/ARC-AGI
cd ARC-AGI/
# Ensure you have at least python 3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp config/api_keys.env.example config/api_keys.env
# Add your keys
set -a && source config/api_keys.env && set +a
python run.py --task-directory tasks_no_answers_single_task/ --answers-directory answers_only_single_task/
```
You should see something like this, and it should complete in a few minutes only

```
Found 1 task files. Total test cases: 1
Starting batch execution with 20 parallel task workers...

  Task         Status       Step                      Outcome   Duration
 ────────────────────────────────────────────────────────────────────────
  2ba387bc:1   🟡 RUNNING   Step 1 (Shallow search)                04:03
```

This task should complete in a few minutes, and since we have specified --answers-directory in the command line the script will also correct the answer. When it finishes, it should look something like this:

```
  Task         Status         Step       Outcome   Duration
 ───────────────────────────────────────────────────────────
  2ba387bc:1   🟢 COMPLETED   Finished   PASS         06:02

Submission file saved to: submissions/2025-11-30_23-55-50_submission.json
```

Do note that --answers-directory is optional. All answers are stores in submissions/ in one big .json file as well as individual answer files.


To run the full eval 2 data set just run the command `python run.py --task-directory tasks_eval2_no_answers/`

... or if you want the script to also correct the answers, then run `python run.py --task-directory tasks_eval2_no_answers/ --answers-directory answers_only_eval2/`

By default, the script has 20 --task-workers, and the output will look something like this:
```
  Task         Status         Step                       Outcome   Duration
 ───────────────────────────────────────────────────────────────────────────
  16de56c4:1   🟡 RUNNING     Step 5 (Full search)                    43:37
  16de56c4:2   🟡 RUNNING     Step 5 (Full search)                    43:37
  20a9e565:1   🟡 RUNNING     Step 5 (Full search)                    43:37
  20a9e565:2   🟡 RUNNING     Step 5 (Full search)                    43:37
  247ef758:1   🟡 RUNNING     Step 5 (Full search)                    43:37
  247ef758:2   🟡 RUNNING     Step 5 (Full search)                    43:37
  332f06d7:1   🟡 RUNNING     Step 5 (Full search)                    43:37
  36a08778:1   🟡 RUNNING     Step 5 (Full search)                    43:37
  36a08778:2   🟡 RUNNING     Step 3 (Extended search)                43:37
  3dc255db:1   🟡 RUNNING     Step 5 (Full search)                    43:37
  581f7754:1   🟡 RUNNING     Step 5 (Full search)                    43:37
  581f7754:2   🟡 RUNNING     Step 3 (Extended search)                43:37
  67e490f4:1   🟡 RUNNING     Step 5 (Full search)                    43:37
  71e489b6:1   🟡 RUNNING     Step 5 (Full search)                    43:37
  71e489b6:2   🟡 RUNNING     Step 5 (Full search)                    43:37
  7b3084d4:1   🟡 RUNNING     Step 5 (Full search)                    43:37
  80a900e0:1   🟡 RUNNING     Step 5 (Full search)                    43:37
  898e7135:1   🟡 RUNNING     Step 5 (Full search)                    43:37
  8b9c3697:1   🟡 RUNNING     Step 1 (Shallow search)                 06:12
  8e5c0c38:1   🟡 RUNNING     Step 1 (Shallow search)                 00:55
  409aa875:1   🟢 COMPLETED   Finished                   PASS         42:42
  5961cc34:1   🟢 COMPLETED   Finished                   PASS         37:25
```

# Overview

## Algorithm

This is mainly an LLM based algorithm across the major frontier models. It does however implement some funky stuff like multimodal data, hint extraction, deeper analysis through the prompt, as well as some degree of smarter searching through the solutions.

I've done a bunch of analysis to come to this algorithm that may be informative to others trying to solve ARC AGI. Please see below for some of that analysis, or just hook me up on the ARC AGI discord channel.

## Current performance

The performance is: 70.7% on arc agi 2 eval dataset

Full results here: https://www.kaggle.com/code/johanland/johan-land-solver-v6-public/output?scriptVersionId=286318109&select=submissions.tgz

The latest additions are:
1. An object / transformation solver: First step to identify all objects in the grids, second step to identify the most obvious transformations, third step to solve based on these as hints. Helps abstract away on noisy problems, e.g. https://arcprize.org/play?task=dbff022c
2. Multi-agent ("ARC LOGIC AUDITOR" and "ARC Solution Auditor") answer judges. Helps find solutions in cases where they are hard to find, and many of the models make "obvious but identical mistakes" giving clusters around false solutions.

## Historical performance

I did a run on Dec 1 on the Eval 2 dataset which yielded a 50.3% solved problems (each sub task measured independently). 

Full results here: https://github.com/beetree/ARC-AGI/blob/main/docs/RESULTS.md

## Next steps

The next improvements that I'm hoping to do to my algorithm are:
- ~~Use the reasoning traces in better choosing and refining the solutions~~
- I suppose I should implement an explicit python based solver too. I don't like the idea of doing the search in "python space" (seems inefficient) but I can see it being very powerful in verifying solutions, or even "implementing the best of solutions"
- In my current algorithm, it seems that the multi modal part of the solver is only marginally adding to the performance, but I want to experiment a bit more with nano banana pro, imagen, sora, veo, etc. I think there is a fundamentally additative form of insights that come from these models
- Grok doesn't seem to be particularly performant, but I guess I should try it out too. I'm more curious about the chinese open source models. They may add completely other angles that could be highly complimentary.
- Test anthropic-beta: effort-2025-11-24, with reasoning high and maybe temperature?
- Try gemini 3 with 2.0 temperature
- Should solve all tests within multi-test tasks in one go with an even deeper search. It'll help validate the answers, and probably also to extend to the harder test cases. For example: https://arcprize.org/play?task=d35bdbdc
- ~~Implement a sparse grid representation as part of the search for solutions~~
- ~~Improve the search within problems by extracting the "most obvious transformation", and then just applying that to all the inputs (including the "test" set) and then solving the problem~~
- ~~There's something broken with my gemini implementation. Google is for sure using something else (other settings, or other model) in their submission to arc agi 2, not sure how to reconcile~~
- ~~Be more conscious around temperature and possibly other model parameters as well~~
- ~~Improve on the hint methodology, there's more performance to gain here~~
- ~~Train a separate model to predict the likelihood of a solution actually being correct, and use this to guide the search. Right now there's just heuristics around this. Need to be conscious not to overfit while doing this.~~

Also some small fixes needed:
- Make hints speak in terms of color-numbers (e.g. #7)rather than explicit colors (e.g. orange)

## Historical analysis

## Objects and transformations

Significant performance increase on hard problems (likely +5-10pp on total) by splitting solvers into three steps:
1. Extract objects
2. Identify key transformations
3. Feed objects and transformations to final solver

On a problem like https://arcprize.org/play?task=dbff022c this significantly changes the performance. ~70% of all models cluster around a solution where they use the "palette" as top-to-bottom instead of inside-out. The object-transformation approach generates the correct solution by providing things like these:

Objects 1: `Objects: Hollow Frames/Enclosures, Interior Fills/Pixels, Multi-colored Clusters, Background (Color 0).\n    Attributes: Color, Shape (rectangular vs irregular), Containment (empty vs filled), Connectivity.`

Objects 2: `The grids contain: (1) rectangular frames/borders of single colors with hollow interiors, (2) multi-colored palette/key grids serving as reference blocks, (3) irregular shapes with non-rectangular or shifted boundaries, (4) small cross/plus patterns, and (5) patterned rectangles with partial fills. Key attributes include color (values 1-9, with 0 as background), size, position, boundary regularity (regular vs. irregular), and interior fill state (hollow, solid, patterned, or multi-colored).`

Transformation 1: `The transformation identifies a multi-colored \"legend\" block to create a color mapping (Key -> Value) used to fill the hollow interiors of shapes. The orientation and reading direction of the legend depend on its aspect ratio and position: Wide blocks map vertically (Top->Bottom if at top, Bottom->Top if at bottom), while Tall blocks map horizontally (Left->Right). Objects with colors found in the 'Key' position of the legend have their enclosed areas filled with the corresponding 'Value' color; others remain unchanged.`

Transformation 2: `1. **Identify palette**: A 2-row (horizontal) or 2-column (vertical) multi-colored grid serving as a lookup table\n2. **Create mapping**: First row/column = frame color keys; Second row/column = fill color values\n3. **For each shape with frame color in palette keys**: Replace all enclosed interior 0s with the mapped fill color\n4. **Shapes with colors NOT in keys**: Remain completely unchanged\n5. **Non-shape elements** (crosses, palette, exterior 0s): Remain unchanged\n6. **Frame colors never change**: Only interior 0s are modified`

As you can see in the objects the concept of a palette is identified, and even the "bottom-top" logic which is key to coming to the right solution (which most models get wrong) is correctly identified.

## Multi agent judge

The key problem that this is overcoming is bad models clustering around wrong answers. For some problems there is an obvious and seemingly correct logic for how to solve it, but in a non-immediately-obvious way this solution can be proven wrong. In practice, this simplifies it quite a bit, but the principle is as simple as this.

The solution (not perfect, but adding significant performance) is to chain another LLM call focused purely on assessing the answers. This is a very large prompt, on order of 30-80k input tokens that takes the full reasoning from each of the models for each of their attempted methodology and then scores them based on two methodologies: (1) attempts to recreate the logic/code that the solution is using and tests its ability of solving the test cases, and (2) checks the multiple solution candidates for their consistency (e.g. they all came to the same answer, but did they do so in the same way). Based on the results of these two judges the two attempted answers are selected.

I've only run this for a set of select problems where it seems to help a lot, but haven't yet tested it for the full dataset - e.g. I haven't yet measured the adverse impact of it, for example it might choose the wrong solutions by "being smart" when the obvious solution is actually the right solution.

Below is an example of a non-sensical reasoning by GPT-5.1-high on this problem: https://arcprize.org/play?task=d35bdbdc

```
...
Applying this to the **test input**:
- I scanned all possible 3×3 windows (top-left positions (r,c) with 0 ≤ r ≤ 7, 0 ≤ c ≤ 7).
- In every 3×3 window, the 8 border cells were **not all the same non-zero color**; there is no 3×3 frame of the kind used in the training examples.
- Therefore, the set of special squares is empty; the graph has no nodes; no paths, no pairings, and thus **no modifications** are applied.
So the output grid is identical to the input grid.
...
```

This is a mistake that several models end up doing, therefore there's a cluster of solutions around bad answers.

### Visual Hints
[Visual Hints Analysis](docs/ANALYZE_VISUAL_HINTS.md)

With explicit but condensed solution instructions there is a material improvement in performance (~5pp) of which maybe half stays even with condensed hints that plausibly could be deduced in some general way.

It is very hard to extract hints that are non-obvious. The observations / transformations that I manage to extract are most of the time only 80% of the solution, and the missing ~20% are the "gotchas" that the models really need to move the needle - they already know "the 80%".

Thereby, it doesn't seem that the hints are truly helping.

### Multi modal, adding images
[Multi modal, adding images Analysis](docs/ANALYZE_MULTIMODAL_IMAGES.md)

The models does some reasoning better based on images. Therefore we should either supply the images directly to the model, or do it in two stages to extract new insights through an image-only prompt that we then supply to a second stage text-only prompt that solves the problem.

Testing supplying the images directly using different types of generated images does not seem to be helpful in solving the harder problems.

### Deep Think Trigger
[Deep Think Trigger Analysis](https://github.com/beetree/ARC-AGI/blob/main/docs/ANALYZE_DEEP_THINK_TRIGGER.md)

It seems that attempting to trigger deeper thinking does not actually yield any deeper thinking. The models themselves already trigger a very deep (deepest possible?) think by themselves.

### Answer Verification (before judge approach)
[Answer Verification Analysis](docs/ANALYZE_VERIFICATION.md)

A key thing to devising the overall approach to solving the problems is to know when the problem has been solved, and ideally do so with high confidence. To do this, I've attempted an approach of replaying test cases with synthetic data expansion selectively. I've done this tuning for precision (e.g. avoiding falsely verifying answer that turn out to be wrong).

When saying something is verified it truly is correct ~94% of the time (precision) on problems that reasonably could be truly solved, while falsely setting them as not verified when they actually were true ~50% of the time (1-recall). On the full sample though, the precision is closer to 90-100%, but the recall drops downwards of 25-30%.

To ensure the precision truly approaches 100% I'm going to add in that the solution needs to be run twice with matching grids together with a perfect score on the test data with synthetic expansion. Later on I'll train a model to actually predict the likelihood of a true PASS based on all the testing done up to then.

### Strategy Extraction
[Strategy Extraction Analysis](docs/ANALYZE_STRATEGY_EXTRACTION.md)

In order to refine the results we need to introduce a concept of "strategy" (or explanation) of why the model has chosen to come to a certain answer/conclusion. This "strategy" works like a guide that can be applied to other test cases thereby enabling us to validate responses, either through testing it on the supplied test data or on synthetic data. Or even by building up a library of strategies from solved problems, generalizing them and mapping them to identifyable traits of problems to supply strategies at solution time for unknown problems.

I tested several approaches of getting a "strategy" out of the model. Many had an actual performance implication by affecting the reasoning, through a distraction or deterioration of the spacial reasoning. In the end, the best proved to be a two stage prompt approach where the first stage outputs the solution, and a second stage outputs the strategy with as much context as possible retained from the first step (same session id, etc).

### Base Model Performance
[Base Model Analysis](docs/ANALYZE_BASE_MODELS.md)

- Overall, similar performance across the smartest versions of OpenAI, Claude and Gemini
- GPT-5.1 with no reasoning is the only viable low latency and low cost model, though its performance is very low. Should only be used for very basic tasks.
- For a slightly smarter model it seems that Sonnet 4.5 with thinking turned off is a good choice. Higher latency and cost, but significant gain in performance. This could be the base solution model, whereas GPT-5.1 could be used as a "parsing model"
- Ideally, something should be stitched together using these models, and then the bigger models as "teachers" (extracting generic strategies, structuring the problem space, etc) and potentially also be used selectively as sanity checks on the very hardest problems
- GPT-5.1 with High reasoning performs the best (note: this is different from the official leaderboard). If perfectly combined with Gemini 3 High and Sonnet 4.5 with maximum thinking budget, around ~half of the errors are eliminated as compared to GPT 5.1 High stand-alone. E.g. there is significant non-overlap in the problems they solve. And, since Gemini 3 High and Sonnet 4.5 max budget are both lower latency, the only consideration in doing all three is cost - assuming the best solution can correctly be identified.

### Grid Format
[Grid Format Analysis](docs/ANALYZE_GRID_FORMAT.md)

- 9 different basic formats were analyzed
- Formats that are not spoken-language-friendly clearly underperform (ascii, compact).
- Regular formats that have clear natural-language-separators perform roughly the same (xml, spaces, comma, semicolon, python) though there are some differences
- There is potential in formats that capture the nature of specific problems, like a sparse representation or a binary color representation, although the performance uplift is small enough that it may not be worth implementing it
- Specifically, sparse does outperform on certain sparse problems (e.g. https://arcprize.org/play?task=1cf80156). The uplift - if perfectly identified - is ~10% so it's not completely negligible, but it might be subsumed by more obvious improvements. Hence, will likely leave it for later
- CSV is the highest performing format, and outperforms the other natural-language-friendly formats by 10-20%. Python and space/newline-separated are the two runner ups but they underperform with 90% significance (sampled across 3x1000 problems)
- Formats not (yet) explored: object representation yet (e.g. islands, matched over rotations), condensed (e.g. 3 black, then 10 blue, ...), input-to-output diff, image/multimodal
- I have not yet explored the performance increase from having multiple representations present at the same time
- Conclusion: For now, will use CSV grid format only, and later on explore problem-specific representations (sparse, object, diff, image, etc)

### Strategy Usage
[Strategy Usage Analysis](docs/ANALYZE_STRATEGY.md)

gpt-5.1-none is pretty much unable of following a strategy, so this will only work for better models.

gpt-5.1-low can reasonably well follow at least a simple strategy if explicit and specific to the problem. When this is the case, it can be lifted to "medium" level. It can't however follow "hints" that are more general

gpt-5.1-medium is very good with explicit strategies that are unique to the problem and is easily lifted to "high" performance if not more. It also can consume generic advice but it likely has to be at least somewhat relevant to the problem.

gpt-5.1-high is already very high performant but with and explicit strategy problems that it previously couldn't solve become solvable. This suggests that the model isn't failing because it has some inherit inabilities (e.g. certain transforms not possible). It likely can also make previously unsolvable problem solvable simply by hints, but the specificity required of these hints requires further research.

