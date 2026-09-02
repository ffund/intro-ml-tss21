import sys
from pathlib import Path


path = Path(sys.argv[1])
if path.name != "1-intro-ml.html":
    raise SystemExit

html = path.read_text()
if "poll-title-slide" in html:
    raise SystemExit

title_open = '<section id="title-slide">'
title_start = html.find(title_open)
if title_start == -1:
    raise RuntimeError(f"Could not find title slide in {path}")

poll = """
<div class="poll-title-content">
  <img class="poll-title-qr" src="../images/polleverywhere-ffund.png" alt="QR code for PollEv.com/ffund">
  <div class="poll-title-copy">
    <div class="poll-title-prompt">Open on your phone/laptop:</div>
    <div class="poll-title-url"><a href="https://pollev.com/ffund" target="_blank" rel="noopener">PollEv.com/ffund</a></div>
    <div class="poll-title-instructions">In the &quot;Name&quot; field put your<br>net ID (e.g. ff524) and<br>&quot;Continue&quot;.</div>
  </div>
</div>
"""

html = html.replace(title_open, '<section id="title-slide" class="poll-title-slide">', 1)
title_start = html.find('<section id="title-slide" class="poll-title-slide">')
title_end = html.find("</section>", title_start)
if title_end == -1:
    raise RuntimeError(f"Could not find end of title slide in {path}")

html = html[:title_end] + poll + html[title_end:]
path.write_text(html)
