# Fonts

The renderer needs a Thai-capable font. It auto-resolves in this order:

1. `assets/fonts/Sarabun-Regular.ttf` (preferred — ships with a standard
   Google Fonts license, widely used in Thai fintech UIs)
2. macOS system fonts: ThonburiUI, Ayuthaya, SukhumvitSet, Silom
3. Explicit `--font` path passed to the CLI

**Download Sarabun for reproducibility:**

```bash
curl -L -o assets/fonts/Sarabun-Regular.ttf \
    https://github.com/cadsondemak/Sarabun/raw/master/fonts/Sarabun-Regular.ttf
curl -L -o assets/fonts/Sarabun-Bold.ttf \
    https://github.com/cadsondemak/Sarabun/raw/master/fonts/Sarabun-Bold.ttf
```

License: SIL Open Font License v1.1 — redistributable.
