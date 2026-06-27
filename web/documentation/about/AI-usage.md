---
title: "AI Usage Disclosure - MCPower"
description: "How generative AI tools were used in building MCPower's engine, ports, and documentation, and how all AI-assisted output was reviewed and validated."
---
# AI usage disclosure

As a developer who use AI to programming I want to be clear about where AI contributed and where it did not. I wrote the core of MCPower entirely myself, and most of it before AI coding tools were available: the project itself, the first MVP, and the demo it grew from. Just as importantly, I developed every concept behind it — the robustness testing, the low-level linear algebra, the choice of statistical methods, and the validation gates that the rest of the project must pass. This is the part that has to be correct, and AI contributed nothing to it. The amount of AI involvement only grows as you move outward from this core, and the sections below describe all of it honestly. The only AI tool I used was Anthropic's Claude: Claude Code for the code and the writing, and Claude Design alongside it for the interface.

## With a little help from AI

For some of the surrounding work, AI acted as an assistant while I made the decisions. It helped with literature search, with writing the validation and benchmarking code, and with debugging. I set the direction and made the judgements; AI reduced the manual effort.

## In an AI–human iterative loop

A large part of MCPower was built in a back-and-forth between me and Claude. This is how the interface and user experience were developed (with Claude Design), together with the WebAssembly and Tauri applications, this documentation and the examples book, the test suites, the release and CI tooling, and the code comments — where AI rewrote and now keeps consistent almost all of the comments throughout the codebase.

## Where AI did most of the work

The largest AI contribution is in porting code, where the target is exact and so leaves little room for judgement. The R package is a full rewrite of my Python port, produced by Claude Code. Likewise, where existing open-source code had to be adapted into MCPower, I let Claude Code do most of the translation with limited supervision: when the goal is to reproduce existing code almost exactly, the standard is precise, and any output that does not match the original closely enough is not used. Wherever such code is adapted, the original authors are credited.

## Verification and responsibility

Every part that AI contributed to was reviewed and went through the same tests and validation as the code I wrote by hand. I take full responsibility for all code in MCPower repository.
