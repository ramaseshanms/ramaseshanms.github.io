# ramaseshanms.github.io

Personal portfolio site for **Ramaseshan Subramanian** — Senior AI Systems Engineer.

Live at: **https://ramaseshanms.github.io**

---

## Architecture

```
ramaseshanms.github.io/
├── index.html              # Main single-page portfolio
│                           #   Sections: Hero, Experience, Projects, Skills, Writing, Contact
├── page.html               # Generic markdown renderer (any .md becomes a page)
├── style.css               # Complete design system (CSS custom properties, no framework)
├── app.js                  # Scroll reveal animations, nav glassmorphism, mobile toggle
│
├── writing/                # DROP .md FILES HERE — each becomes a standalone page
│   └── vulkan-llm-runtime.md
│
├── blogs/                  # Legacy folder (backward compat — old blog URLs redirect here)
│   └── FMA-Net++.md
│
├── images/
│   └── profile.jpg         # Hero profile photo (square, min 500x500 recommended)
│
├── resume.pdf              # Linked from hero "Resume" button (add manually, not in git)
│
├── blog.html               # Redirect shim: old ?file= URLs → new page.html?slug= format
├── technical.html          # Redirect → index.html#projects
├── script.js               # Legacy (empty — functionality moved to app.js)
├── blog.js                 # Legacy (empty)
├── blogs-list.js           # Legacy (empty)
└── blog-loader.js          # Legacy (empty)
```

**Zero build step.** No SSG, no npm, no bundler. Push HTML/CSS/JS to `main` and GitHub Pages serves it.

---

## How to Add New Writing

### Step 1 — Write the markdown

Create a `.md` file in `writing/`. Use any filename — it becomes the URL slug.

```
writing/my-new-post.md  →  page.html?slug=my-new-post
```

Start the file with a top-level heading and optional author/date line:

```markdown
# Your Post Title

*Author Name | Month 2026*

---

Content starts here. Full GitHub-flavored markdown supported:
tables, code blocks, images, links, etc.
```

### Step 2 — Add a card to index.html

Open `index.html` and find the `<!-- Writing -->` section. Add a new `<a>` block inside `<div id="writing-list">`:

```html
<a href="page.html?slug=my-new-post" class="writing-card reveal">
  <div class="writing-card__date">2026-04-15</div>
  <div class="writing-card__title">
    Your Post Title
    <span class="writing-card__arrow">&rarr;</span>
  </div>
  <div class="writing-card__excerpt">
    One or two sentences describing what the post is about.
  </div>
</a>
```

Keep newest posts at the top.

### Step 3 — Push

```bash
git add writing/my-new-post.md index.html
git commit -m "writing: add my-new-post"
git push
```

Live within ~30 seconds at `https://ramaseshanms.github.io/page.html?slug=my-new-post`.

---

## How page.html Works

`page.html` is a generic markdown renderer. It:

1. Reads `?slug=` from the URL query string
2. Fetches `writing/{slug}.md` (falls back to `blogs/{slug}.md` for old posts)
3. Extracts the `# Title` and optional `*author*` line from the markdown
4. Renders the rest via [marked.js](https://cdn.jsdelivr.net/npm/marked/marked.min.js) (loaded from CDN)

No registry file needed. If the `.md` exists and the card links to it, it works.

---

## Design System

### Stack
- **HTML/CSS/JS** — vanilla, no frameworks
- **Font**: [Inter](https://fonts.google.com/specimen/Inter) (body) + [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono) (code)
- **Icons**: Unicode characters (no icon library)
- **Markdown rendering**: [marked.js](https://marked.js.org/) via CDN (page.html only)

### Color Palette (CSS custom properties in `style.css`)

| Variable | Value | Usage |
|----------|-------|-------|
| `--bg` | `#0a0a0b` | Page background |
| `--bg-raised` | `#111113` | Cards, alternating sections |
| `--bg-surface` | `#18181b` | Tags, pills, inputs |
| `--border` | `#27272a` | Default borders |
| `--text` | `#fafafa` | Primary text |
| `--text-secondary` | `#a1a1aa` | Body paragraphs |
| `--text-muted` | `#71717a` | Dates, labels |
| `--accent` | `#818cf8` | Links, highlights (indigo-400) |
| `--accent-glow` | `rgba(129,140,248,0.15)` | Hover glows |

### Key Design Decisions

1. **Dark theme only** — matches the terminal/systems engineering identity. No light mode toggle (intentional — this is a portfolio for low-level engineers, not a SaaS landing page).

2. **Scroll reveal animations** — elements fade up (30px translate + opacity) on viewport entry via `IntersectionObserver`. Class: `reveal`. Add `reveal-delay-1` through `reveal-delay-4` for staggered entry.

3. **Frosted glass nav** — transparent on page load, gains `backdrop-filter: blur(20px)` + border on scroll (class `scrolled` toggled by `app.js`).

4. **Gradient hero title** — `linear-gradient(135deg, indigo → purple → pink)` with `background-clip: text`. The "fast on real hardware" is the gradient portion.

5. **Subtle grid background** — CSS `background-image` with `mask-image` radial fade on the hero section. No image file needed.

6. **Timeline for experience** — vertical line with dots, not cards. Cleaner for 2-3 roles. If you add more roles, the pattern scales naturally.

7. **Project cards with stats bar** — each card has tags + a bottom stats row with highlighted metrics. The `::before` pseudo-element adds a gradient top border on hover.

8. **Skill pills** — individual skills as rounded pills with hover glow. Grouped by category in 3-column grid.

9. **Writing cards** — minimal list style (not cards). Indent-on-hover with arrow reveal. Prioritizes scanability over decoration.

10. **Article typography** — `max-width: 800px`, generous line-height (1.8), styled tables with hover rows, code blocks with monospace font and border. Designed for technical content with lots of tables and code.

### Responsive Breakpoints

- **768px**: Hero stacks vertically (photo on top), nav collapses to hamburger, grids go single-column
- **480px**: Reduced padding, smaller headings

---

## Updating Content

### Change profile photo
Replace `images/profile.jpg`. Square aspect ratio, minimum 500x500px. Grayscale filter applied via CSS (lifts on hover).

### Update resume
Drop `resume.pdf` in the root directory. The hero button links to it directly.

### Add/edit experience
Edit the `<!-- Experience -->` section in `index.html`. Each role is a `.timeline__item` with a `.timeline__dot`, header, and bullet list.

### Add/edit projects
Edit the `<!-- Projects -->` section. Each project is a `.project-card` in the `.grid-2` container. Copy an existing card and modify.

### Add/edit skills
Edit the `<!-- Skills -->` section. Each group is a `.skill-group` with `.skill-pill` spans inside.

---

## Legacy URL Compatibility

Old URLs are redirected automatically:

| Old URL | Redirects to |
|---------|-------------|
| `blog.html?file=blogs/FMA-Net++.md` | `page.html?slug=FMA-Net++` |
| `technical.html` | `index.html#projects` |

The `blogs/` folder is kept for backward compatibility. `page.html` checks `writing/` first, then falls back to `blogs/`.

---

## Deployment

**Platform**: GitHub Pages (automatic on push to `main`)
**Domain**: `ramaseshanms.github.io` (default GitHub Pages domain)
**HTTPS**: Automatic via GitHub
**Build**: None — static files served directly
