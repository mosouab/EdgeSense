# EdgeSense pitch deck (web)

3-minute pitch, in French, on the Université d'Évry "Think It & Do It" (JITT) template.
Plain HTML/CSS/JS, no build step, no network: open `index.html` in Chrome or Firefox.

## Before presenting

Fill in the two dashed-yellow placeholders on slide 1 (`Prénom NOM` for the team and the tutor)
in `index.html`, then remove the `fill-me` class. Delete the tutor line if there is none.
The speaker notes say "Je m'appelle [Prénom]" on slide 1 too.

## Presenting

| Key | Action |
|---|---|
| `→` `Space` `PageDown` | next slide (clickers work) |
| `←` `PageUp` | previous slide |
| `F` | fullscreen |
| `P` | presenter window: notes, next slide, 3:00 pace timer |
| `B` | black screen (mirrored to the audience window) |
| `T` / `R` | timer start-pause / reset (presenter window) |
| `H` | shortcut help |

Put the main window on the projector (fullscreen), press `P`, and keep the presenter window on
the laptop screen. Both windows stay in sync whichever one you drive. The timer starts on the
first slide change; each slide has a target end time (`data-duration` on each `<section>`).
`index.html#5` opens directly on slide 5.

## PDF backup

`EdgeSense-pitch.pdf` is the same deck, one slide per page, for a USB stick. Regenerate it after
editing (uses the system Google Chrome):

```
uv run --no-project --with playwright python pitch/web/export.py
uv run --no-project --with playwright python pitch/web/export.py --png /tmp/slides --presenter   # + PNG per slide
```

## Where the numbers come from

- **4/4 documented failures, ~10 days between false alarms**: event-level view of
  `reports/full_evaluation/scores_timeline.csv` over the 139-day evaluation region, counting only the
  4 official Metro.PT failures (not the 2 audit-added events). 15 alarm episodes, 11 of them outside
  any official failure (one every ~12.6 days); the slide rounds down to ~10 to stay conservative.
- **< 1 ms per window on one core, 0.17 MB / 41.5k parameters**: `docs/EDGE_BENCHMARK.md`
  (0.48–0.56 ms median on a 1-core Pi-class proxy, not physical Pi hardware).
- **Dashboard screenshots**: the real simulator driven into the Metro.PT failure #3 alert
  (`pitch/build/capture_dashboard.py` in the working copy), i.e. replayed real data, stated on the slide.

## Credits

Template backgrounds extracted from `Template Pitch - JITT 19 04.pdf`. Fonts: Archivo and IBM Plex
Mono (SIL OFL). Icons: Lucide (ISC), inlined as an SVG sprite. Licences are in `fonts/`.
