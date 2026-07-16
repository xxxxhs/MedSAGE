# Reproducibility Notes

Default settings: retrieval top-K = 50; FAISS search K = 80; sentence selection top-K = 10; τ = 0.7; C:R:D = 1:1:1; segmentation regex = `re.split(r"[。！？!?；;]+", text)`; min length = 4; max length = 180. Retrieval-level leakage control removes records with the same question ID before evidence construction.
