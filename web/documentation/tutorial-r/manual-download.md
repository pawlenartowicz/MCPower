---
title: "MCPower for R - Manual Binary Downloads"
description: "Direct download links for prebuilt MCPower R binaries - Windows, macOS, and Linux, for the current and previous R release - when the one-line install does not fit your R version."
---
# Manual binary downloads

For almost everyone the one-line install is all you need — it pulls a prebuilt
binary for your platform, no compiler required:

```r
install.packages("mcpower", repos = "https://r.mcpower.app")
```

This page is the fallback for the cases that one-liner can't cover automatically.
Use it when:

- you are on **Linux with the previous R release** (the one-liner serves the
  *current* R release for Linux; older R needs the matching file below), or
- you want to **pin a specific build**, or install **offline** from a saved file.

## Pick the file that matches your R version

A prebuilt binary is tied to the R **minor** version it was built for — a build
for R 4.5 needs R 4.5.x, a build for R 4.4 needs R 4.4.x. Check yours first:

```r
getRversion()
#> [1] '4.5.1'      # → use an R4.5 file below
```

Then install straight from the link (no compiler, no Rust toolchain):

### Windows

```r
# R 4.5 (current release)
install.packages("https://r.mcpower.app/download/windows/mcpower-R4.5.zip", repos = NULL)
# R 4.4 (previous release)
install.packages("https://r.mcpower.app/download/windows/mcpower-R4.4.zip", repos = NULL)
```

### macOS (Apple Silicon)

```r
# R 4.5 (current release)
install.packages("https://r.mcpower.app/download/macos/mcpower-R4.5.tgz", repos = NULL)
# R 4.4 (previous release)
install.packages("https://r.mcpower.app/download/macos/mcpower-R4.4.tgz", repos = NULL)
```

### Linux (x86_64)

Built on AlmaLinux 8 (glibc 2.28), so it loads on any current Linux distribution.

```r
# R 4.5 (current release)
install.packages("https://r.mcpower.app/download/linux/mcpower-R4.5.tar.gz", repos = NULL)
# R 4.4 (previous release)
install.packages("https://r.mcpower.app/download/linux/mcpower-R4.4.tar.gz", repos = NULL)
```

## Nothing here fits — build from source

If you are on a platform or R version with no prebuilt binary (Intel Mac, a
32-bit system, an older R, or musl-libc Linux), compile the current source. This
needs a [Rust toolchain](https://www.rust-lang.org/tools/install):

```r
remotes::install_github("pawlenartowicz/mcpower", subdir = "ports/r", build = FALSE)
```

Back to the [[tutorial-r/index|R tutorial]].
