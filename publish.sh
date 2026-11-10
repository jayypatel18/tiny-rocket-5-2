#!/bin/sh
# Publish the film: create the public GitHub repo, push, turn on GitHub Pages, check the URL.
# Jay runs this himself (the Claude session is not allowed to create a public repo on its own).
# Safe to run again: each step is skipped or repeated harmlessly if it was already done.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="jayypatel18/tiny-rocket-5-2"

if ! gh repo view "$REPO" >/dev/null 2>&1; then
  gh repo create "$REPO" --public \
    --description "Tiny Rocket: a 74 second anime short where every frame and every sound is code written by Fable 5.2" \
    --source "$ROOT" --remote origin --push
else
  git -C "$ROOT" remote get-url origin >/dev/null 2>&1 || git -C "$ROOT" remote add origin "git@github.com:$REPO.git"
  git -C "$ROOT" push -u origin main
fi

# GitHub Pages from /docs on main
gh api -X POST "repos/$REPO/pages" -f "source[branch]=main" -f "source[path]=/docs" >/dev/null 2>&1 \
  || gh api -X PUT "repos/$REPO/pages" -f "source[branch]=main" -f "source[path]=/docs" >/dev/null 2>&1 || true
gh api -X PATCH "repos/$REPO" -f homepage="https://jayypatel18.github.io/tiny-rocket-5-2/" >/dev/null 2>&1 || true

echo "Waiting for Pages to build..."
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "https://jayypatel18.github.io/tiny-rocket-5-2/")
  film=$(curl -s -o /dev/null -w "%{http_code}" -r 0-99 "https://jayypatel18.github.io/tiny-rocket-5-2/film.mp4")
  echo "page $code, film $film"
  if [ "$code" = "200" ] && { [ "$film" = "200" ] || [ "$film" = "206" ]; }; then
    echo "LIVE: https://jayypatel18.github.io/tiny-rocket-5-2/"
    echo "Repo: https://github.com/$REPO"
    exit 0
  fi
  sleep 15
done
echo "Pages did not answer 200 yet. Check https://github.com/$REPO/settings/pages"
exit 1
