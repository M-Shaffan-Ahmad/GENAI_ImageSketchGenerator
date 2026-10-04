# Task 4 paired sketch evaluation

Split test; pairs 1068; epoch 53; smoke False.

Targets are deterministic synthetic pencil, ink and charcoal approximations, not human-drawn sketch annotations. Blank white baseline helps expose background-driven scores. Pairwise generated-style L1 versus target-style L1 measures output diversity, not style correctness. Inspect the same photograph across all three generated styles and the separate target/error panels. Failure candidates use distinct source photos. Test evaluation is separate and does not run in the supplied Colab notebook.
