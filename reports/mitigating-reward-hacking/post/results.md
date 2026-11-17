<div class="blk" markdown="1">
<p class="lede-l">Setup</p>

Each Harbor run used one A100 (80 GB) and a four-hour wall-clock limit. We disabled web search and access to other runs.

The launcher trains the three development environments concurrently on the shared GPU, with a two-hour limit per environment per training attempt. Final evaluation uses the same per-environment training limit.
</div>

{{table:results}}

<div class="blk" markdown="1">
<p class="lede-l">Held-out reference solution</p>

We ran GRAFT's published training code and configuration with seed 1 and scored its models with the same grader.
</div>

{{figure:passk}}

{{figure:tradeoff}}

{{figure:transfer}}
