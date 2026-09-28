---
applyTo: "**/*.{ts,html,css,scss,dart}"
---

# UI/UX Pro Max Design Rules

Apply these rules whenever creating or modifying UI code.

## Priority 1 — Accessibility (CRITICAL)
- Minimum contrast 4.5:1 for text; 3:1 for large text
- Visible focus rings on all interactive elements
- Descriptive alt text for meaningful images
- aria-label for icon-only buttons/controls
- Tab order matches visual reading order
- Respect `prefers-reduced-motion` in animations
- Don't convey info by color alone — add icon or text

## Priority 2 — Touch & Interaction (CRITICAL)
- Min 44×44pt (iOS) / 48×48dp (Android/Web) touch targets
- Min 8px gap between adjacent touch targets
- Disable submit buttons during async operations; show loading state
- Use platform-standard gestures; never block system gestures

## Priority 3 — Performance (HIGH)
- Use WebP/AVIF images with srcset; lazy-load below-fold images
- Declare image width/height to prevent layout shift (CLS < 0.1)
- Split bundles by route; lazy-load non-critical components
- Skeleton screens for operations > 1 second

## Priority 4 — Style Selection (HIGH)
- Use SVG icon sets (Heroicons, Lucide) — never emoji as UI icons
- Keep one consistent style across all pages
- Match style to product type: developer tools → dark + terminal; SaaS → glass + slate; fintech → blue + minimal
- Only one primary CTA per screen; secondary actions visually subordinate

## Priority 5 — Layout & Responsive (HIGH)
- Mobile-first: design 375px first, scale up to 768/1024/1440
- No horizontal scroll on mobile
- Max container width: max-w-6xl or max-w-7xl on desktop
- Viewport meta: `width=device-width, initial-scale=1` (never disable zoom)

## Priority 6 — Typography & Color (MEDIUM)
- Base body font-size 16px minimum; line-height 1.5–1.75
- Semantic color tokens (primary, error, surface) — no raw hex in components
- Heading/body font pairing (e.g. Space Grotesk + DM Sans)
- Weight hierarchy: headings 600–700, body 400, labels 500

## Priority 7 — Animation (MEDIUM)
- Duration 150–300ms micro, ≤400ms complex
- Animate transform/opacity only (never width/height/top/left)
- Ease-out for entering elements; ease-in for exiting
- Respect `prefers-reduced-motion: reduce`

## Priority 8 — Forms & Feedback (MEDIUM)
- Visible label on every form field (not placeholder-only)
- Error messages positioned near the problem field
- Progressive disclosure: don't overwhelm with optional fields upfront
