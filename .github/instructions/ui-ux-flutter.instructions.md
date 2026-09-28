---
applyTo: "flutter/**/*.dart"
---

# UI/UX Pro Max — Flutter Design Rules

Apply these rules when creating or modifying any Flutter widget, screen, or theme.

## Widgets (workern_widgets)
- Always check `flutter/packages/workern_widgets/lib/widgets/` before building custom UI.
  App bars → `WorkernAppBar`, Star rating → `StarReview` (star_review.dart).
- Follow Material 3 — use `Theme.of(context).colorScheme` tokens, not hardcoded colors.
- Use `const` constructors wherever possible to prevent unnecessary rebuilds.

## Touch Targets (CRITICAL)
- Min 48×48dp for all tap targets (Material guideline).
- Min 8dp gap between adjacent tappable elements.
- Wrap small icons in `IconButton` to meet target size automatically.

## Loading & Async (CRITICAL)
- Disable buttons during async ops — set `onPressed: null` or show inline `CircularProgressIndicator`.
- Use `Shimmer` / skeleton widgets for lists that take > 300ms to load.
- Never show a blank screen while data is loading — always show a loading state.

## Accessibility
- All meaningful images must have `semanticsLabel`.
- Icon-only buttons must have `Semantics(label: '...')` or `Tooltip`.
- Support Dynamic Type — use `Theme.of(context).textTheme` scale, never hardcode font sizes.
- Honour `MediaQuery.of(context).disableAnimations` for reduced-motion.

## Navigation
- Use GoRouter; routes are defined in `lib/config/router.dart`.
- Never navigate with raw `Navigator.push` — use `context.go()` or `context.push()`.
- Bottom navigation max 5 items; use `NavigationBar` (Material 3).

## State & Riverpod
- Loading states via `AsyncValue`: handle `.when(data:, loading:, error:)` explicitly.
- Never show an empty screen on error — show an error widget with retry action.
- Invalidate providers after write operations: `ref.invalidate(provider)`.

## Typography & Color
- Use `Theme.of(context).textTheme` — `titleLarge`, `bodyMedium`, `labelSmall`.
- ANTAR ecosystem colors: primary matches the app's brand color (saffron/brown/blue etc.).
- Dark/light theme: define both in `AppTheme`; never hardcode `Colors.white` or `Colors.black`.

## Layout
- Use `SafeArea` on all top-level screens.
- Use `SingleChildScrollView` + `ConstrainedBox` for forms to avoid overflow on small screens.
- Horizontal scroll on phone is a UX bug — redesign with vertical or wrap layout.

## Animation
- Duration 150–250ms for micro-interactions; use `Curves.easeOut` entering, `Curves.easeIn` exiting.
- Prefer `AnimatedSwitcher`, `AnimatedContainer`, `Hero` over manual `AnimationController`.
- Always check `MediaQuery.of(context).disableAnimations` and skip animation if true.
