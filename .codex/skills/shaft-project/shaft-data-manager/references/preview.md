# Preview Policy

Use previews to inspect annotation quality, not as the source of truth.

## Defaults

- Do not regenerate previews unless the user asks.
- For normal bbox inspection, draw boxes directly on the original image with controlled line
  thickness and label size.
- Avoid zoom panels unless the user explicitly needs local inspection.
- Keep temporary previews in a temp directory when investigating suspicious cases.

## Validation Use

- Use previews for a small set of representative or suspicious samples before large rebuilds.
- For known bad cases, produce a separate temporary preview folder rather than mixing them into
  the main preview directory.
- Do not let preview outputs drive annotation changes without checking the raw JSON.
