# Remember base
git merge-base master not-astro
ce7b0b5337a9c52822f5f88a61cec1cc7ddbe39e

# Save all changes from not-astro to files
git diff ce7b0b5337a9c52822f5f88a61cec1cc7ddbe39e > full-diff.txt
git diff ce7b0b5337a9c52822f5f88a61cec1cc7ddbe39e --compact-summary > summary-diff.txt
