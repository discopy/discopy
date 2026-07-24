# TODO — Implement issue #462

User prompts, verbatim:

> implement this issue now [https://github.com/discopy/discopy/issues/462](https://github.com/discopy/discopy/issues/462)

> wait a second, take your time to do it properly

## Checklist

- [WIP] @codex-2026-07-24 20:06 Replace the separate drawing fixture workflow with documentation images
      as executable drawing baselines, including safe compare/replace behavior,
      migrated examples, concise tests, and full verification.

## Mathematical description

A documented drawing is a deterministic map from a diagram and drawing
parameters to an image. When an image already exists, drawing should check that
the map still returns the same baseline; explicit replacement should update the
baseline. Thus documentation examples become the single source of truth for
both presentation and regression testing.
