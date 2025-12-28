# Conclusion

...


# Background

The models seem to be using python internally, but it doesn't seem consistent. Maybe by implementing support for tool calls they'll perform better?

OpenAI:
```
    "tools": [
        {
            "type": "code_interpreter",
            "container": {"type": "auto"}  # optionally add: "memory_limit": "4g"
        }
    ],
    "include": ["code_interpreter_call.outputs"],
    "max_tool_calls": 10,
    "tool_choice": "auto",
```

Gemini:
```
    tools=[types.Tool(code_execution=types.ToolCodeExecution)],
```


# Performance

Ran the frontier 18: 3dc255db:1,8b7bacbf:2,0934a4d8:1,e3721c99:2,62593bfd:1,88e364bc:1,20a9e565:2,e3721c99:1,a25697e4:2,3a25b0d8:2,7b0280bc:1,eee78d87:1,269e22fb:1,4e34c42c:1,78332cb0:1,7b80bb43:1,89565ca0:1,a32d8b75:1

Without tools:
- gemini-3-high: 1 PASS out of 36 attempts (6 timed out / crash). Errors: 429 RESOURCE_EXHAUSTED
- gpt-5.2-xhigh: 5 PASS out of 72 attempts (24 timed out / crash). Errors: 7 timeout, 11 token, 6 unknown/support

With tools:
- gemini-3-high: 5 PASS out of 36 attempts (0 timed out / crash). Errors:
- gpt-5.2-xhigh: 1 PASS out of 72 attempts (13 timed out / crash). Errors: 6 timeout, 2 token, 5 unknown/support

Gemini seems to significantly outperform with tools, whereas gpt is outperforming without tools. This is an interesting insight. Furthermore, there seems to be fewer crashes/time-outs when using tools.

