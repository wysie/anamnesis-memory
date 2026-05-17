# Anamnesis Dashboard Visual System

Status: proposed direction for the local admin/dashboard UI.

## Research takeaways

Sources reviewed:

- Updivision, “UI Color Trends to Watch in 2026” — color systems should be adaptive, accessible, and distinctive; elevated neutrals reduce eye strain; dark and light modes should feel like one coherent brand rather than two unrelated skins.
- AILogoCreator, “Logo Design Trends 2026” — strong marks are moving toward high-contrast minimal geometry, bold wordmarks, responsive identity variants, and subtle human texture rather than generic AI-perfect gloss.
- Untitled UI, “28 Best Free Fonts for Modern UI Design in 2026” — modern UI fonts should optimize screen legibility first; open/free families such as Inter, Public Sans, Geist, IBM Plex, Space Grotesk, DM Sans, and JetBrains Mono are safe practical choices for product UIs.
- Impeccable (`https://impeccable.style/`, `pbakaus/impeccable`) — product UI should feel trusted and familiar, not strange for flavour; use OKLCH and tinted neutrals, avoid pure black/white, avoid “AI slop” such as purple gradients, gradient text, card nesting, thick side borders, decorative glassmorphism, modals by reflex, redundant copy, and Inter-only typography.
- Internal design-system references for dark developer dashboards: precise dark surfaces, sparse chromatic accents, high hierarchy, strong mobile hamburger navigation, readable metadata chips, and right-side detail drawers.

## Brand personality

Anamnesis should feel like:

- local-first memory infrastructure
- calm, precise, private, and auditable
- more “instrument panel / archive console” than “AI chatbot”
- premium but understated
- trustworthy enough for sensitive memory governance

Avoid:

- neon cyberpunk
- loud generic AI gradients
- cartoon brain imagery
- over-mystical copy
- pure black + pure white harshness
- unexplained architecture slogans

## Recommended default theme: dark operational console

Use dark as the primary dashboard theme because memory review, audit, and debugging are operational workflows. Light mode can exist later, but dark should be the canonical visual identity.

### Color tokens

Implementation tokens should be authored in OKLCH, with hex comments only if needed for handoff. Neutrals are slightly violet-tinted instead of pure gray; do not use pure `#000` or `#fff` for large UI surfaces.

```css
:root {
  color-scheme: dark;

  --bg: oklch(12% 0.012 268);             /* near-black violet navy */
  --bg-elevated: oklch(15% 0.014 268);
  --surface: oklch(18% 0.016 268);
  --surface-2: oklch(22% 0.018 268);
  --surface-3: oklch(27% 0.020 268);

  --text: oklch(96% 0.006 268);
  --text-muted: oklch(75% 0.018 268);
  --text-subtle: oklch(58% 0.020 268);
  --text-faint: oklch(42% 0.018 268);

  --border-subtle: oklch(28% 0.018 268);
  --border: oklch(34% 0.020 268);
  --border-strong: oklch(44% 0.026 268);

  --accent: oklch(68% 0.145 279);
  --accent-2: oklch(72% 0.135 305);
  --accent-soft: oklch(24% 0.055 279);
  --accent-border: oklch(54% 0.090 279);

  --success: oklch(72% 0.130 158);
  --warning: oklch(78% 0.140 78);
  --danger: oklch(70% 0.160 21);
  --info: oklch(76% 0.115 225);

  --gold: oklch(76% 0.095 82);
  --cream: oklch(94% 0.020 77);
}
```

### Color roles

- Background: near-black navy, not pure black. Gives depth without eye strain.
- Main accent: periwinkle/violet. Use for selected nav, primary buttons, active filters, and focus rings.
- Gold: use rarely for “canonical / trusted / accepted” states, logo sparkle, or audit milestones.
- Green: active/accepted/safe.
- Amber: pending/inbox/review-needed.
- Red: rejected/invalidated/destructive.
- Cyan: technical metadata, source/platform provenance, diagnostics.

## Light theme direction

Light mode should use elevated neutrals, not stark white.

```css
[data-theme="light"] {
  color-scheme: light;

  --bg: oklch(96% 0.018 77);
  --bg-elevated: oklch(98% 0.014 77);
  --surface: oklch(99% 0.008 77);
  --surface-2: oklch(93% 0.018 77);
  --surface-3: oklch(89% 0.024 77);

  --text: oklch(20% 0.012 268);
  --text-muted: oklch(45% 0.016 77);
  --text-subtle: oklch(62% 0.018 77);
  --text-faint: oklch(74% 0.016 77);

  --border-subtle: oklch(90% 0.020 77);
  --border: oklch(84% 0.026 77);
  --border-strong: oklch(76% 0.030 77);

  --accent: oklch(50% 0.155 279);
  --accent-2: oklch(54% 0.145 305);
  --accent-soft: oklch(92% 0.035 279);
  --accent-border: oklch(68% 0.075 279);
}
```

## Typography

Use a three-font system:

1. UI/body: Geist Sans or system UI stack
   - Best default: Geist Sans if we want a more modern/product-engineered feel.
   - Safe fallback: `-apple-system`, BlinkMacSystemFont, `Segoe UI`, system UI.
   - Avoid an Inter-only system. Inter is acceptable as a fallback, but “Inter everywhere” is a recognizable generic AI-design tell.
   - Body 14–16px, line-height 1.45–1.6. Use at least 16px for mobile body text.

2. Display/section headings: Space Grotesk
   - Gives Anamnesis a stronger identity than generic Inter-only dashboards.
   - Use for page titles, stat numerals, empty-state headlines, and brand wordmark.
   - Keep weights 500–700; avoid huge hero marketing typography inside admin screens.

3. Metadata/code: JetBrains Mono or Geist Mono
   - Use for rids, source labels, platform scopes, timestamps, CLI snippets, audit event names.
   - Keep small: 11–13px.

Recommended CSS:

```css
:root {
  --font-ui: "Geist", -apple-system, BlinkMacSystemFont, "Segoe UI", ui-sans-serif, system-ui, sans-serif;
  --font-display: "Space Grotesk", "Geist", ui-sans-serif, system-ui, sans-serif;
  --font-mono: "JetBrains Mono", "Geist Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}

body {
  font-family: var(--font-ui);
  font-size: 14px;
  line-height: 1.5;
  letter-spacing: -0.01em;
}

h1, h2, h3, .brand-wordmark, .metric-value {
  font-family: var(--font-display);
  letter-spacing: -0.035em;
}

code, .rid, .timestamp, .event-type, .chip-mono {
  font-family: var(--font-mono);
}
```

## Logo direction

Do not use a brain icon. It is too generic for memory tools.

Recommended logo system:

1. Primary mark: “A” monogram as an archive knot
   - Construct from a geometric capital A.
   - Add one small orbit/dot or folded-corner notch to imply recall/linkage.
   - Must work at favicon size.

2. Wordmark: “Anamnesis” in Space Grotesk Medium/Semibold
   - Tight but readable tracking.
   - Use a small gold or violet dot/accent as the distinctive mark.

3. Responsive variants
   - Full lockup: icon + Anamnesis wordmark.
   - Sidebar compact: icon only.
   - Favicon: simplified monogram only.
   - Loading/empty-state: monogram inside a soft ring.

Logo colors:

- Dark theme icon: `#f4f6fb` on `#08090c`, accent dot `#7c8cff` or `#d0aa63`.
- Light theme icon: `#15171d` on `#f7f4ef`, accent dot `#6657d9`.

## Core dashboard layout

Use these patterns:

- Desktop: left sidebar + top utility row + card/grid content.
- Mobile: compact header with brand + hamburger. Menu hidden by default.
- Memory browser: responsive card grid, not permanent split-pane.
- Memory detail: right-side drawer on desktop, near-fullscreen drawer on mobile.
- Review/inbox: queue cards with strong lifecycle badges and bulk actions.
- Audit: timeline component with event chips, reason text, and linked rids.
- Recall simulator: two-column result view on desktop; stacked cards on mobile.

## Component styles

### Cards

Avoid card nesting. Use cards only for distinct memory records, queue items, or independent panels. Inside a card, use typography, spacing, and subtle dividers instead of another card.

```css
.card {
  background: linear-gradient(180deg, var(--surface-2), var(--surface));
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 18px 60px color-mix(in oklch, var(--bg) 72%, transparent), inset 0 1px 0 var(--border-subtle);
}
```

### Buttons

- Primary: violet fill, white text, 10–12px radius.
- Secondary: translucent surface, subtle border.
- Destructive: red outline/fill only where needed.
- Toolbar buttons: compact, mono/icon optional.

### Badges

Lifecycle badge colors:

- active / accepted: green
- pending / inbox: amber
- rejected: muted red
- invalidated: gray/red outline
- corrected: violet/gold
- source/platform: cyan/blue-gray

### Detail drawer

Desktop:

```css
.drawer {
  position: fixed;
  top: 22px;
  right: 22px;
  width: min(760px, calc(100vw - 44px));
  max-height: calc(100vh - 44px);
  border-radius: 24px;
  overflow: auto;
}
```

Mobile:

```css
@media (max-width: 720px) {
  .drawer {
    inset: 72px 10px 10px 10px;
    width: auto;
    max-height: none;
    border-radius: 20px;
  }
}
```

## Impeccable guardrails

Apply these before any dashboard implementation or polish pass:

- Register: this is product UI, not brand/marketing UI. Design serves memory review and audit tasks.
- Scene sentence: Primary user is reviewing private memory state on desktop or phone, often quickly, with sensitive/local data. The UI should feel calm, precise, and trustworthy, not theatrical.
- Use OKLCH tokens and tinted neutrals. Avoid pure black/white and avoid raw alpha-heavy palettes except focus rings or intentional overlays.
- Choose a color strategy first. Default is restrained: neutral surfaces plus one accent under roughly 10% visual weight, with semantic colors only for state.
- Use a 4pt spacing scale: 4, 8, 12, 16, 24, 32, 48, 64.
- Fixed rem type scale for dashboard UI, not fluid marketing typography.
- Every interactive element needs default, hover, focus-visible, active, disabled, loading, error, and success treatment where applicable.
- Prefer skeleton states over centered spinners.
- Prefer inline/progressive disclosure over modals. Use drawers/pages for complex memory details and settings.
- Use specific verb-object labels: “Accept memory”, “Reject proposal”, “Correct memory”, “Keep editing”, not “OK”, “Submit”, or “Yes”.
- Motion should convey state only: 100–150ms for feedback, 200–300ms for menus/tooltips, 300–500ms for drawers; use quart/quint/expo easing; respect `prefers-reduced-motion`.
- Mobile is structural: hamburger navigation, safe-area support, 44px touch targets, no hover-only functionality.

### Anti-slop checklist

Reject these patterns during implementation:

- purple-to-blue gradients as generic AI decoration
- gradient text
- decorative glassmorphism and blur cards
- cards inside cards
- thick coloured side borders on rounded cards
- identical icon-card grids
- huge icon tiles above every heading
- hero metric template as the main overview pattern
- Inter everywhere with no typographic contrast
- low-contrast gray text on coloured backgrounds
- redundant copy that repeats heading, subheading, helper, and hint
- modals as first thought for complex workflows
- bounce/elastic motion

## Information hierarchy

Order of priority:

1. What is the memory/proposal?
2. Is it active, pending, rejected, invalidated, corrected?
3. Who owns it and where can it recall?
4. Where did it come from?
5. Why was it saved/rejected/corrected?
6. What action can I safely take?

Avoid showing raw database-like metadata before the human-readable text/status.

## Copy tone

Dashboard copy should be plain and operational:

Good:
- “Pending memory review”
- “This item is not recallable until accepted.”
- “Corrected from previous memory”
- “Source platform: WhatsApp”
- “Rejected as low-value chat fragment”

Avoid:
- “Cognitive substrate activated”
- “Memory consciousness graph”
- “AI brain state”
- “One embedded engine, many indexes”

## Implementation notes

- Self-host fonts/assets for local/private dashboard use.
- Cache-bust CSS/JS during active mobile testing.
- Verify desktop and 390–430px mobile screenshots before calling UI work done.
- Keep routes and actions deep-linkable where possible.
- No native `prompt()`/`alert()` for admin actions; use in-app modals.
- Do not expose the local dashboard over a public tunnel unless explicitly requested.

## Initial design decision

Recommended starting point: dark operational console with Space Grotesk + Inter + JetBrains Mono, periwinkle/violet primary accent, rare gold trust accent, and a geometric Anamnesis “A” monogram.

This direction is distinctive enough to avoid generic SaaS, but still conservative enough for a memory-governance/admin tool.
