# Glossary

- **AUT (area under time):** TESSERACT's normalised area under a per-period performance curve, summarising decay over a deployment window.
- **Base rate:** the real-world prevalence of the positive class; drives precision at deployment.
- **Calibration set:** held-out data used to compute conformal p-values; never used for training.
- **Concept drift:** change in data distribution over time.
- **Confusable (homoglyph):** a character that looks like another, e.g. Cyrillic "а" and Latin "a".
- **Conformal prediction:** framework for calibrated confidence and rejection.
- **Feature-space attack:** modifying feature vectors directly.
- **Functionality check:** a test that a transformed sample still behaves like the original.
- **Held-out transformation family:** a class of evasions excluded from training and used only to test robustness.
- **Monotonic classifier:** score cannot decrease when certain features increase.
- **NFKC:** Unicode compatibility normalisation that folds many look-alike forms into canonical characters.
- **Overlay:** data appended to a PE file beyond its declared sections.
- **Problem-space attack:** modifying a real object (binary, email) while preserving semantics.
- **Temporal split:** training on past, testing on future data.
- **Zero-width character:** an invisible Unicode code point (e.g. U+200B) that can split tokens.
