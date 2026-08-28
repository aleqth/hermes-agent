# Inline image request sanitization

Hermes keeps image-bearing turns in durable conversation history, but provider
requests must not replay every historical screenshot on every call.

Immediately after the exact provider/MoA request is assembled and before token
estimation or the pre-spend budget gate, Hermes now rebuilds only the outbound
request copy. It keeps the newest unique inline images within the active
route's `max_inline_images` and `max_inline_image_bytes` rails. Exact duplicates
and older overflow images are omitted from that request. Persisted messages and
session history are not mutated.

If the newest single image is itself larger than the byte rail, it remains in
the request copy so the existing hard budget guard still blocks the provider
call. The sanitizer therefore prevents historical replay fan-out without
weakening fail-closed payload protection.

The implementation supports OpenAI-style data URLs, Anthropic base64 image
parts, and Gemini `inline_data` parts. MoA prepared requests receive the same
sanitized message list used by the budget gate and aggregator.

Regression coverage proves that five unique images are reduced to the newest
four, nine repeated images are reduced to one, the original history is
unchanged, the byte cap prefers the newest images, and an individually
oversized newest image remains blocked before provider spend.
