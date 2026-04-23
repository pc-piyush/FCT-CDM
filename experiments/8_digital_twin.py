import torch
import copy
import numpy as np

def simulate_counterfactual(model, d, c, t, intervention_idx=None, intervention_type="drug_boost"):

    """
    Simple digital twin:
    Modify early trajectory → observe risk change
    """

    model.eval()

    print("[DT] Baseline prediction...")

    base = model(d, c, t).item()

    d_cf = copy.deepcopy(d)
    c_cf = copy.deepcopy(c)
    t_cf = copy.deepcopy(t)

    print("[DT] Applying intervention:", intervention_type)

    # -----------------------------
    # Intervention 1: increase drug exposure
    # -----------------------------
    if intervention_type == "drug_boost":

        for i in range(len(d_cf)):

            if d_cf[i] == 1:  # drug domain
                c_cf[i] = c_cf[i] + 100  # artificial shift

    # -----------------------------
    # Intervention 2: remove early severe events
    # -----------------------------
    if intervention_type == "early_event_removal":

        for i in range(min(10, len(d_cf))):
            d_cf[i] = 0
            c_cf[i] = 0

    print("[DT] Counterfactual prediction...")

    cf = model(d_cf, c_cf, t_cf).item()

    print(f"[DT] baseline={base:.4f}, counterfactual={cf:.4f}")

    return base, cf