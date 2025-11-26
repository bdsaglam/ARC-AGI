# Strict JSON output
Requiring both an `explanation` and the `grid` - both as free text.
Enforced through the API call requiring strict JSON output (OpenAI) following this format
This greatly underperforms (gpt-5.1-none), due to the loss of spatial reasoning (e.g. 1,3,4\n1,1,0\n4,6,0, the introduction of "\n")

# XML formatting
<explanation/> and <grid/>
Still underperforms greatly (5.2% vs 10.0%)

# Asking for the reasoning summary
For OpenAI (and Gemini? And something from Claude too?) it's possible to get a reasoning summary
This seems to be free, in the sense that there is no performance drop (10.1% vs 10.0%)
The quality of these reasoning summaries is dubious. It's more of hints to what actually happened. Likely it's helpful in reconstructing the actual strategy, but they couldn't be used off-the-shelf as they come out of the API.

# Two step call, grid output only first, then explanation
No impact on performance (as should be, 10.4% vs 10.0%)
Very high quality explanation/strategy, customizable through the prompt, and it might even carry over some of the reasoning from the previous step (e.g. the grid-only output) through the preserved session id - though this is quite unclear if it's actually happening.

# Two step call, explanation first, then grid output only
This seems to drastically drop the performance. I've tried changing the prompts but it seems to just inately be bad (6.7% vs 10.0%)



