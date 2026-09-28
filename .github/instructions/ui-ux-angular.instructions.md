---
applyTo: "angular/**/*.{ts,html,css,scss}"
---

# UI/UX Pro Max — Angular Design Rules

Apply these rules when creating or modifying any Angular component, template, or style.

## Components (spartan/ui)
- Always use `angular/packages/spartan/{component}` before creating custom UI.
  Buttons → `hlm-button`, Cards → `hlm-card`, Inputs → `hlm-input`,
  Dialogs → `hlm-alert-dialog`, Selects → `hlm-select`, Badges → `hlm-badge`
- Use Tailwind classes; never write custom CSS for spacing or color.
- Use CSS variable tokens (`text-foreground`, `bg-muted`, `text-primary`) — no raw hex.

## Accessibility (CRITICAL)
- Every interactive element must be keyboard-focusable with a visible 2px ring.
- Icon-only buttons must have `aria-label`.
- Contrast minimum 4.5:1 — test both light and dark variants.
- `[tabindex]` must match visual reading order.

## Touch & Loading States (CRITICAL)
- Min 44×44px click targets.
- Disable buttons during async ops; show inline spinner via signal:
  `protected readonly isLoading = signal(false);`
- Always show skeleton or spinner for data that takes > 300ms.

## Templates
- Use `@if`, `@for`, `@switch` (not `*ngIf`/`*ngFor`).
- Use `async` pipe for streams; never subscribe in templates.
- Show empty-state UI when lists are empty — not just nothing.
- Error messages must appear near the relevant field, not only at top.

## Forms
- Visible label on every field — never placeholder-only labels.
- Reactive forms only; bind `[formControl]` directly.
- Show validation errors on blur or submit, not on every keystroke.

## Typography & Color
- Body 16px min; line-height 1.5; use `text-sm`/`text-base`/`text-lg`.
- Heading scale: use Tailwind `text-xl`/`text-2xl`/`text-3xl` consistently.
- Font pairing for ANTAR ecosystem: Inter (body) + Space Grotesk (headings)
  (or the app-specific font set in its `styles.css`).

## Responsive
- Mobile-first: design for 375px, scale up.
- Use `max-w-6xl mx-auto` for page containers.
- Horizontal scroll on mobile is a blocking bug.

## Animation
- Transitions: 150–200ms ease-out for micro, ≤400ms for complex.
- Use CSS `transition` on `opacity` and `transform` only.
- Add `@media (prefers-reduced-motion: reduce)` guards for all animations.
