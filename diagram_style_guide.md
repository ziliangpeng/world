# Architecture Diagram Style Guide

## Visual Style Analysis (from vllm_architecture.jpg)

### Color Palette
- **Cyan/Teal gradient**: `#5BCCC9 → #6B9BD6` (API Server, Tokenizer)
- **Blue/Purple gradient**: `#6B8FD6 → #9B7BD6` (Engine Core - central component)
- **Orange/Coral gradient**: `#E89B7A → #F4B799` (Scheduler, Block Manager - grouped subsystem)
- **Purple/Lavender**: `#B87BD6 → #C99BD6` (Sampler)
- **Green/Teal**: `#4A8B6B → #5AA87B` (Workers/Executors)

### Design Elements
- **Rounded corners**: All boxes have significant border radius (~20-30px)
- **Soft shadows**: Subtle drop shadows for depth (offset ~5-10px, blur ~15-20px, low opacity)
- **Gradient fills**: Smooth linear gradients within each box
- **Clean typography**: Sans-serif, bold for component names, lighter weight for descriptions
- **Two-line structure**: Component name (bold) + description (smaller, lighter)
- **Container grouping**: Light rounded rectangle container for related components (Scheduler + Block Manager)

### Connector Style
- **Smooth curves**: Bezier curves instead of straight lines
- **Arrows**: Simple solid arrowheads
- **Labels**: Small text on connectors describing the relationship
- **Color**: Neutral gray or matching component colors
- **Width**: Medium stroke width (~3-4px)

### Layout
- **Hierarchy**: Top-down flow (API Server → Engine Core → downstream components)
- **Central focus**: Engine Core larger and centered
- **Grouped subsystems**: Visual containers for tightly-coupled components
- **Balanced spacing**: Generous whitespace between components
- **Light background**: White or very light gray

---

## Reusable Prompt Template

Use this template when asking AI to create architecture diagrams in the same style:

```
Create a modern, high-quality architecture diagram with the following components and relationships:

[COMPONENTS - List your components here with their roles]

[RELATIONSHIPS - Describe connections between components]

Style Requirements (IMPORTANT - match existing style):
- **Color palette**:
  - Cyan/teal gradients (#5BCCC9 → #6B9BD6) for entry/utility components
  - Blue/purple gradients (#6B8FD6 → #9B7BD6) for central/core components
  - Orange/coral gradients (#E89B7A → #F4B799) for grouped subsystems
  - Purple/lavender (#B87BD6 → #C99BD6) for processing components
  - Green/teal (#4A8B6B → #5AA87B) for execution components

- **Box style**:
  - Rounded corners (border-radius: ~25px)
  - Soft drop shadows (offset 5-10px, blur 15-20px, low opacity)
  - Smooth linear gradient fills
  - Two-line text: bold component name + lighter description below

- **Typography**:
  - Clean modern sans-serif (like SF Pro, Inter, or Helvetica)
  - Bold for component names
  - Regular weight for descriptions
  - Good contrast against gradient backgrounds

- **Connectors**:
  - Smooth Bezier curves (not straight lines)
  - Medium stroke width (~3-4px)
  - Simple solid arrowheads
  - Small labels on arrows describing relationships
  - Neutral colors or matching component colors

- **Layout**:
  - Top-down flow for hierarchy
  - Central component should be larger and centered
  - Group related components in light rounded containers
  - Generous whitespace
  - Balanced, clean composition

- **Overall vibe**:
  - Modern, sleek, professional
  - Apple/Stripe/Vercel design aesthetic
  - Minimalist but polished
  - High contrast for readability
  - Premium tech documentation quality
```

---

## Quick Reference

When creating new diagrams:
1. Copy the template above
2. Fill in [COMPONENTS] and [RELATIONSHIPS] sections
3. Keep all style requirements intact
4. Use color palette based on component type:
   - **Cyan/Teal**: Entry points, utilities
   - **Blue/Purple**: Core/central components
   - **Orange/Coral**: Grouped subsystems
   - **Purple**: Processing/logic components
   - **Green**: Execution/computation components
