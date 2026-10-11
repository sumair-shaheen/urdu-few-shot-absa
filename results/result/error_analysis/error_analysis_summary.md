# Error Analysis Summary

All results pooled across 3 seeds at 20-shot.

| Model | Method | N examples | Accuracy | N misclassified | Negation-flagged | Code-mix-flagged |
|---|---|---|---|---|---|---|
| mbert | Full Fine-Tune | 5067 | 0.4462 | 2806 | 485 | 5 |
| mbert | LoRA | 5067 | 0.4174 | 2952 | 505 | 7 |
| xlmr | Full Fine-Tune | 5067 | 0.5735 | 2161 | 386 | 5 |
| xlmr | LoRA | 5067 | 0.5786 | 2135 | 381 | 6 |
| mt5 | Full Fine-Tune | 5067 | 0.3856 | 3113 | 543 | 6 |
| mt5 | LoRA | 5067 | 0.3300 | 3395 | 435 | 9 |
| mt5_incontext | In-Context Prompting | 5067 | 0.3333 | 3378 | 663 | 6 |

*Negation/code-mixing flags are simple keyword heuristics meant to surface candidate examples for manual reading -- they are not a validated linguistic error taxonomy. Always read the actual sentence before citing it as an example of a specific error type.*