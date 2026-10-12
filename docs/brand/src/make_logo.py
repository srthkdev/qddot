"""Compose the qddot ASCII hero banner and icon as SVG."""
from html import escape
from pathlib import Path

robot = Path("docs/brand/robot.txt").read_text().rstrip("\n").split("\n")
title = [
    "   ◆   ◆",
    " ██████╗ ██████╗ ██████╗  ██████╗ ████████╗",
    "██╔═══██╗██╔══██╗██╔══██╗██╔═══██╗╚══██╔══╝",
    "██║   ██║██║  ██║██║  ██║██║   ██║   ██║",
    "██║▄▄ ██║██║  ██║██║  ██║██║   ██║   ██║",
    "╚██████╔╝██████╔╝██████╔╝╚██████╔╝   ██║",
    " ╚══▀▀═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝",
]
spell = "q̈ = M(q)⁻¹ · ( τ − C(q, q̇) q̇ − g(q) )"
tagline = ["run · check · debug robot code in physics", "before your agent says it's done"]

FONT = "ui-monospace, 'JetBrains Mono', SFMono-Regular, Menlo, Consolas, monospace"
DEFS = """<defs>
  <linearGradient id="metal" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#ffb35c"/><stop offset="0.55" stop-color="#5fd4ff"/><stop offset="1" stop-color="#2a6f9e"/>
  </linearGradient>
  <linearGradient id="ink" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#7fe0ff"/><stop offset="1" stop-color="#c9f2ff"/>
  </linearGradient>
  <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
    <feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <radialGradient id="bg" cx="0.3" cy="0.4" r="0.9">
    <stop offset="0" stop-color="#101826"/><stop offset="1" stop-color="#05070b"/>
  </radialGradient>
</defs>"""


def block(lines, x, y, size, fill, lh=1.0, extra=""):
    spans = "".join(f'<tspan x="{x}" dy="{0 if i == 0 else size * lh:.1f}">{escape(l) or " "}</tspan>'
                    for i, l in enumerate(lines))
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
            f'xml:space="preserve" {extra}>{spans}</text>')


def frame(w, h, m=14):
    c = "#3a4a63"
    ticks = "".join(f'<line x1="{x}" y1="{m}" x2="{x}" y2="{m + 5}" stroke="{c}"/>' for x in range(m + 40, w - m, 40))
    return (f'<rect x="{m}" y="{m}" width="{w - 2 * m}" height="{h - 2 * m}" fill="none" stroke="{c}" stroke-width="1" stroke-dasharray="2 6"/>'
            + "".join(f'<path d="M{x0} {y0 + 18 * sy} V{y0} H{x0 + 18 * sx}" stroke="#ff9f43" stroke-width="2" fill="none"/>'
                      for x0, y0, sx, sy in ((m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)))
            + ticks)


def hero(path):
    w, h = 1280, 560
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="qddot">',
             DEFS, f'<rect width="{w}" height="{h}" fill="url(#bg)"/>', frame(w, h),
             block(robot, 60, 62, 10.4, "url(#metal)", 1.0, 'filter="url(#glow)"'),
             block(title, 560, 150, 21, "url(#ink)", 1.0, 'filter="url(#glow)"'),
             block([spell], 562, 345, 22, "#ff9f43", 1.0, 'font-style="italic"'),
             block(tagline, 562, 410, 19, "#9fb3c8", 1.45),
             block(["// physics is the test suite"], 562, 495, 14, "#4c5d78")]
    Path(path).write_text("".join(parts) + "</svg>\n")


def icon(path):
    s = 256
    glyph = ["  ◆  ◆", " ▄████▄", "██▀  ▀██", "██    ██", "██▄  ▄██", " ▀███▀██", "      ██", "      ▀▀"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {s} {s}" role="img" aria-label="qddot">', DEFS,
             f'<rect width="{s}" height="{s}" rx="48" fill="url(#bg)"/>', frame(s, s, 12),
             block(glyph, 52, 58, 25, "url(#ink)", 1.02, 'filter="url(#glow)"')]
    Path(path).write_text("".join(parts) + "</svg>\n")


hero("docs/brand/hero.svg")
icon("docs/brand/icon.svg")
