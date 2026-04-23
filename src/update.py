def update_patient_tensor(existing_tensor, new_events):
    for domain, events in new_events.items():
        existing_tensor[domain].extend(events)
    return existing_tensor


def add_new_patient(tensor_store, pid, tensor):
    tensor_store[pid] = tensor
    return tensor_store