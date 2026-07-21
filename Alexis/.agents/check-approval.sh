#!/usr/bin/env bash
# check-approval.sh — mechanical INTEGRITY + EXPIRY check for one approval.
# Spec: ROUTINE.md "Approval" section; this script is the binding implementation.
#
# Usage:   check-approval.sh <comment-url> [rocket|code]
#   <comment-url>  a GitHub comment permalink, either
#                  .../pull/N#issuecomment-<id> or .../pull/N#discussion_r<id>
#   rocket (default)  mode (G): comment approved by a :rocket: from ALEXIS_GH
#   code              mode (C): comment is an unedited "/code" spec by ALEXIS_GH
#
# Prints one line — APPROVED / VOID / EXPIRED with the reason — and exits:
#   0 APPROVED   1 VOID   2 EXPIRED   3 usage or API error
# Only exit 0 authorizes implementation. Requires: gh (authenticated), jq.
set -euo pipefail

ALEXIS_GH="${ALEXIS_GH:-toumix}"
APPROVE_EMOJI="${APPROVE_EMOJI_GH:-rocket}"
EXPIRY_DAYS="${EXPIRY_DAYS:-7}"

url="${1:?usage: check-approval.sh <comment-url> [rocket|code]}"
mode="${2:-rocket}"

repo="$(sed -nE 's|^https://github\.com/([^/]+/[^/]+)/.*|\1|p' <<< "$url")"
case "$url" in
    *#issuecomment-*)  api="repos/$repo/issues/comments/${url##*#issuecomment-}" ;;
    *#discussion_r*)   api="repos/$repo/pulls/comments/${url##*#discussion_r}" ;;
    *) echo "ERROR: unrecognized comment URL: $url"; exit 3 ;;
esac
[ -n "$repo" ] || { echo "ERROR: cannot parse repo from URL: $url"; exit 3; }

comment="$(gh api "$api")" || { echo "ERROR: cannot fetch $api"; exit 3; }
created="$(jq -r .created_at <<< "$comment")"
updated="$(jq -r .updated_at <<< "$comment")"
author="$(jq -r .user.login <<< "$comment")"

to_s() { date -u -d "$1" +%s; }
age_days() { echo $(( ($(date -u +%s) - $(to_s "$1")) / 86400 )); }

if [ "$mode" = code ]; then
    [ "$author" = "$ALEXIS_GH" ] \
        || { echo "VOID: /code author is $author, not $ALEXIS_GH"; exit 1; }
    jq -e '.body | startswith("/code")' <<< "$comment" > /dev/null \
        || { echo "VOID: body does not start with /code"; exit 1; }
    [ "$created" = "$updated" ] \
        || { echo "VOID: /code comment edited after creation (at $updated)"; exit 1; }
    age="$(age_days "$created")"
    [ "$age" -lt "$EXPIRY_DAYS" ] \
        || { echo "EXPIRED: /code is ${age}d old (limit ${EXPIRY_DAYS}d)"; exit 2; }
    echo "APPROVED: /code by $ALEXIS_GH, unedited, ${age}d old"
    exit 0
fi

reacted="$(gh api "$api/reactions" --paginate \
    | jq -r --arg u "$ALEXIS_GH" --arg e "$APPROVE_EMOJI" \
        '[.[] | select(.content == $e and .user.login == $u)]
         | sort_by(.created_at) | last | .created_at // empty')"
[ -n "$reacted" ] \
    || { echo "VOID: no :$APPROVE_EMOJI: from $ALEXIS_GH on this comment"; exit 1; }
if [ "$created" != "$updated" ] && [ "$(to_s "$updated")" -ge "$(to_s "$reacted")" ]; then
    echo "VOID: comment edited at $updated, not before the :$APPROVE_EMOJI: at $reacted"
    exit 1
fi
age="$(age_days "$reacted")"
[ "$age" -lt "$EXPIRY_DAYS" ] \
    || { echo "EXPIRED: :$APPROVE_EMOJI: is ${age}d old (limit ${EXPIRY_DAYS}d)"; exit 2; }
echo "APPROVED: :$APPROVE_EMOJI: from $ALEXIS_GH at $reacted, ${age}d old"
exit 0
