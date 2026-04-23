import torch
import numpy as np

def explain_events(model, d, c, t):

    model.eval()

    base = model(d, c, t).item()

    print("[EXPLAIN] baseline:", base)

    importance = []

    for i in range(len(d)):

        d2 = d.clone()
        c2 = c.clone()
        t2 = t.clone()

        d2[i] = 0
        c2[i] = 0

        score = model(d2, c2, t2).item()

        impact = abs(base - score)

        importance.append(impact)

        print(f"[EVENT {i}] impact={impact:.6f}")

    return base, importance