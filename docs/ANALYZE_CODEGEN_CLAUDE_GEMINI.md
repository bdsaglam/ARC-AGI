# Conclusion

# Context

Most of my testing has been with gpt-5.2. Let's now see how it performs with claude and gemini. Let's use the baseline we have for V1B (gpt-5.2-low, using problems 247ef758:1,31f7f899:1,7c66cb00:1,136b0064:1,16de56c4:1,36a08778:1,1818057f:1,38007db0:2,bf45cf4b:1,b0039139:2,1ae2feb7:1,7ed72f31:2,b5ca7ac4:1):
- V1B: 1818057f,bf45cf4b,7c66cb00
- V1B: 1818057f,31f7f899,bf45cf4b,b0039139
- V1B: 1818057f,38007db0,136b0064
- V1B: 1818057f,1ae2feb7
- V1B: 1ae2feb7,1818057f,38007db0,31f7f899,247ef758
- V1B: 1818057f,247ef758,7c66cb00
- V1B: 1818057f,247ef758,bf45cf4b
- V1B: 1818057f,1ae2feb7,16de56c4
- V1B: 1818057f,31f7f899,136b0064
- V1B: 1818057f,247ef758
- V1B: 1818057f,247ef758,b0039139
- V1B: 1818057f,bf45cf4b

I'll run the same problems for claude-opus-4.5-thinking-4000:
- 1818057f
- 38007db0,1818057f
- NA
- NA
- 1818057f
- 38007db0

So, it seems to be somewhat underperforming, but then this is an arbitrary thinking level. At least, it runs without any major errors.

Now let's try gemini-3-low:
- 1818057f,38007db0,bf45cf4b,7ed72f31,

Ok, both models are working. That's good.


# Is there even a point in using Opus and Gemini? Do they add something?

Let's run {a few} very hard but solvable problems on max reasoning for all three models and compare their performance
