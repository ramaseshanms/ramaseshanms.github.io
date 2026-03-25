# ramaseshanms.github.io

Personal portfolio — **Ramaseshan Subramanian**, Senior AI Systems Engineer.

Live at: **https://ramaseshanms.github.io**

---

## Repo Structure

```
ramaseshanms.github.io/
├── index.html          # Single-page portfolio (Hero, Experience, Projects, Skills, Writing, Contact)
├── page.html           # Markdown renderer — fetches writing/{slug}.md and renders it
├── style.css           # Design system (CSS custom properties, no framework)
├── app.js              # Scroll reveal, frosted-glass nav, mobile toggle
├── writing/            # All articles/posts — drop .md here, it becomes a page
│   ├── vulkan-llm-runtime.md
│   └── FMA-Net++.md
├── images/
│   └── profile.jpg     # Hero photo (square, min 500x500)
└── resume.pdf          # Linked from hero button (add manually)
```

**Zero build step.** No SSG, no npm, no bundler. Push to `main` → GitHub Pages serves it.

---

## Adding New Writing

### 1. Create the markdown file

```
writing/my-post-slug.md
```

Start with a heading and optional author line:

```markdown
# Your Post Title

*Author Name | April 2026*

---

Your content here. Full GFM supported: tables, code blocks, images.
```

### 2. Add a card to index.html

Find `<div id="writing-list">` in the Writing section and add at the top:

```html
<a href="page.html?slug=my-post-slug" class="writing-card reveal">
  <div class="writing-card__date">2026-04-15</div>
  <div class="writing-card__title">
    Your Post Title
    <span class="writing-card__arrow">&rarr;</span>
  </div>
  <div class="writing-card__excerpt">
    One or two sentences about the post.
  </div>
</a>
```

### 3. Push

```bash
git add writing/my-post-slug.md index.html
git commit -m "writing: add my-post-slug"
git push
```

Live at `ramaseshanms.github.io/page.html?slug=my-post-slug` within ~30s.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dark theme only** | Matches systems/low-level engineering identity. No light toggle. |
| **Scroll reveal** | Elements fade up (`.reveal` class) via IntersectionObserver. Stagger with `.reveal-delay-1` to `-4`. |
| **Frosted nav** | Transparent at top, gains `backdrop-filter: blur(20px)` on scroll. |
| **Gradient hero text** | `linear-gradient(indigo → purple → pink)` on "fast on real hardware." |
| **Timeline for experience** | Vertical line + dots. Scales cleanly for 2-5 roles. |
| **Project cards with stats** | Tags + bottom metric bar. Gradient top border on hover. |
| **Skill pills with hover glow** | Rounded, grouped by category in 3-col grid. |
| **Writing as list, not cards** | Indent-on-hover + arrow reveal. Prioritizes scanability. |
| **One content folder** | All markdown lives in `writing/`. One place, one pattern. |

## Color Palette

| Variable | Value | Usage |
|----------|-------|-------|
| `--bg` | `#0a0a0b` | Page background |
| `--bg-raised` | `#111113` | Cards, alternating sections |
| `--bg-surface` | `#18181b` | Tags, pills |
| `--border` | `#27272a` | Borders |
| `--text` | `#fafafa` | Primary text |
| `--text-secondary` | `#a1a1aa` | Body text |
| `--text-muted` | `#71717a` | Dates, labels |
| `--accent` | `#818cf8` | Links, highlights (indigo-400) |

## Fonts

- **Body**: [Inter](https://fonts.google.com/specimen/Inter) (400/500/600/700/800)
- **Code**: [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono) (400/500)
- **Markdown**: [marked.js](https://marked.js.org/) via CDN (loaded only on page.html)

---

## Updating Other Content

| What | Where |
|------|-------|
| Profile photo | Replace `images/profile.jpg` (square, CSS applies grayscale filter) |
| Resume | Drop `resume.pdf` in root |
| Experience | Edit `.timeline__item` blocks in `index.html` |
| Projects | Edit `.project-card` blocks in `index.html` |
| Skills | Edit `.skill-group` blocks in `index.html` |
| Contact links | Edit `.contact-row` in `index.html` |
| Metrics bar | Edit `.metrics` div in hero section |
