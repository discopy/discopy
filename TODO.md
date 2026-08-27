# TODO

> oops you're right I forgot the total is accepted+decline+open
> when i said X+Y I meant write the total not literally the string with +
> otherwise looks great!
>
> can you make sure we hit the cache of the style reviewer model ie that each
> prompt we send it is append only from the previous ones

- [ ] The tally counts every remark, the total written as a number: `N style
      remarks taken into account: X accepted / Y declined / Z still open`
- [ ] The prompt is ordered stable-first, so what a round shares with the one
      before it is one prefix: instructions, `STYLE.md`, context, past remarks,
      changed files
- [ ] The numbered list of past remarks only ever grows at its end, the replies
      it drew moving out of it into their own section
- [ ] The budget spends on the changed files first, then context, then the
      remarks, so a growing history cannot evict a context file from the prefix
