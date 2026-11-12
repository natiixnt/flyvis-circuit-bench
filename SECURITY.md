# Security policy

This is a research benchmark: it reads local measurement files and, with `fcb download`, fetches one archive
over HTTPS from the GitHub Release of this repository. The download is verified against published sha256 sums
and extracted only as regular files inside one directory (no links, no absolute paths, no `..`).

## Supported versions

| version | supported |
|---|---|
| 1.0.x | yes |

## Reporting a vulnerability

Please do not open a public issue. Use GitHub private vulnerability reporting ("Report a vulnerability" in the
Security tab of the repository). You can expect a first answer within 14 days.
