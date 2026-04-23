import numpy as np

def get_time_bin(days, bin_size=7):
    # weekly bins by default
    return days // bin_size


def bin_tensor(tensor, bin_size=7):
    binned = {}

    for domain, events in tensor.items():
        new_events = []

        for e in events:
            t = e[0]
            bin_t = get_time_bin(t, bin_size)

            if len(e) == 2:
                new_events.append((bin_t, e[1]))
            else:
                new_events.append((bin_t, e[1], e[2]))

        binned[domain] = new_events

    return binned