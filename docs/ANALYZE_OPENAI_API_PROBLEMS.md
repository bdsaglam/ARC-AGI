# Conclusion

# Background
For the hardest problems, 72% (83 out of 116) of the API calls to OpenAI fails. The failures brake down into three categories:

~20% of errors: `Error during execution: OpenAI Job failed: OpenAI Server Error: Code: server_error, Message: An error occurred while processing your request. You can retry your request, or contact us through our help center at help.openai.com if the error persists. Please include the request ID wfr_019b5b66259b7a73b2ece70572358eed in your message..`

~40% of errors: `Error during execution: OpenAI Job failed: Timeout after 3600s.`

~40% of errors: `Error during execution: OpenAI Job failed: Token limit: IncompleteDetails(reason='max_output_tokens').`

In my experience, the timeouts end up being one of the other two errors even if settings a 7200s timeout. Hence, we have a real problem. And, a good chunk of it is also indetermined (e.g. "contact OpenAI support", which I've done and they have successfully ignored me, as well as given me completely irrelevant and off-the-shelf-generalized suggestions on how their API works). => We're on our own



