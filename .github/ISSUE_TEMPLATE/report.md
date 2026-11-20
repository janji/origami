---
name: Problem or Mac report
about: Something went wrong, or Origami ran on a Mac that is not in the README's table
---

**What happened, and what did you expect?**


**`origami report`** (run it while `origami serve` is running and paste the whole block)

```
paste here
```

**How you start Origami** (`origami serve` options, or your llama-swap entry)


**Client or agent framework**, and the `max_tokens` and `reasoning_effort` it sends


**For a stall:** the output of `curl -s 127.0.0.1:11439/health` while it happens

**For a wrong answer:** the request, or the smallest one that still shows it, at `temperature` 0
