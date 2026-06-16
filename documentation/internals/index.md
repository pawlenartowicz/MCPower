# What's inside MCPower

Every power question you ask MCPower is answered by running thousands of small
simulations: the engine builds synthetic datasets from your design, analyses each
one, and counts how often the effect shows up. The part that makes that fast,
reproducible, and *identical* whether you run it from Python, R, the desktop app,
or the browser is a single compiled **native engine** doing all the numerical work.

You never touch the engine directly — the package or app in front of you is a thin
layer over it. These pages lift the lid for the curious reader who wants to know
how the numbers are produced and why they can be trusted.

- [[internals/engine-architecture|How it's built]] — one engine, four ports, so a
  result in Python is the same calculation as in R, the app, or the browser.
- [[internals/optimizations|Why it's fast and reproducible]] — automatic
  multi-core simulation, a reproducible random-number stream, and the one knob you
  actually control: simulation count versus precision.
- [[validation/index|How we know it's right]] — the evidence that the engine
  generates the design you asked for and solves it the way a trusted reference does.

For the bigger picture of what MCPower is and who it's for, see [[about/index|About MCPower]].
