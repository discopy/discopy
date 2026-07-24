# TODO — Address the final review on PR #470

User prompt, verbatim:

> last round of comments [discopy/discopy#470](https://github.com/discopy/discopy/pull/470)

## Checklist

- [ ] Address all four unresolved review threads: rename the save helper,
      replace regex SVG filtering with structured XML normalization, restore
      the feedback image name, document baseline deletion, and verify fully.

## Mathematical description

Drawing comparison is equality after a canonical projection of an image:
volatile platform-specific SVG representation is discarded structurally while
element order and text content are preserved. A save operation either creates
this baseline, explicitly replaces it, or compares the new canonical image
against it.
