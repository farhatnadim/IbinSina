# Early vs Late Fusion in Multi-Slide MIL

When a patient has multiple slides (e.g., multiple biopsy cores), you need to decide **how to combine them** before making a prediction.

## The Problem

```
Patient A has 3 slides:
  Slide 1: 200 patches → [200, 1024]
  Slide 2: 150 patches → [150, 1024]
  Slide 3: 300 patches → [300, 1024]

How do we get one prediction for Patient A?
```

---

## Early Fusion (`concat_by`)

**Merge first, then attend.**

Concatenate all patches from all slides into one big bag. The model doesn't know which patch came from which slide.

```
Slide 1: [200, D] ─┐
Slide 2: [150, D] ─┼─→ Concatenate → [650, D] → Attention → Patient embedding
Slide 3: [300, D] ─┘
```

**Pros:**
- Simple - works with any standard MIL model
- More patches = more signal

**Cons:**
- Loses slide boundaries
- Can't learn "this slide is more important than that slide"

**Use when:**
- Slides are arbitrary splits of the same tissue
- Slide identity doesn't matter clinically

```python
dataset = MILDataset(labels, features).concat_by('case_id')
# Returns: features [M_total, D] - one tensor per patient
```

---

## Late Fusion (`group_by`)

**Attend first, then merge.**

Keep slides separate. First aggregate patches within each slide, then aggregate slides within the patient.

```
Slide 1: [200, D] → Patch Attention → slide_embed_1 ─┐
Slide 2: [150, D] → Patch Attention → slide_embed_2 ─┼─→ Slide Attention → Patient embedding
Slide 3: [300, D] → Patch Attention → slide_embed_3 ─┘
```

**Pros:**
- Preserves slide structure
- Model can learn which slides matter most
- Interpretable: "Slide 2 drove the prediction"

**Cons:**
- Requires hierarchical model architecture
- More complex batching

**Use when:**
- Slides represent distinct samples (different biopsy sites)
- You want slide-level interpretability
- Clinical relevance of individual slides matters

```python
dataset = MILDataset(labels, features).group_by('case_id')
# Returns: features List[[M_i, D]] - list of tensors per patient
```

---

## Visual Summary

```
EARLY FUSION                          LATE FUSION
─────────────                         ───────────

[Slide 1 patches]                     [Slide 1 patches]
[Slide 2 patches]  → Concatenate      [Slide 2 patches]
[Slide 3 patches]        ↓            [Slide 3 patches]
                    [All patches]           ↓    ↓    ↓
                         ↓             [Attn] [Attn] [Attn]
                    [Attention]             ↓    ↓    ↓
                         ↓             [emb1] [emb2] [emb3]
                    [Patient emb]                ↓
                                           [Attention]
                                                ↓
                                           [Patient emb]
```

---

## Quick Decision Guide

| Question | Early | Late |
|----------|-------|------|
| Are slides from the same tissue region? | ✓ | |
| Do slides represent different biopsy sites? | | ✓ |
| Need slide-level interpretability? | | ✓ |
| Using a standard MIL model? | ✓ | |
| Building a hierarchical model? | | ✓ |
