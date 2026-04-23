import pickle

class VocabBuilder:
    def __init__(self):
        self.token_to_id = {}
        self.counter = 1  # 0 reserved for padding

    def fit(self, tensors):
        print("[VOCAB] Building vocabulary...")

        for pid, tensor in tensors.items():
            for domain in tensor:
                for event in tensor[domain]:

                    if len(event) == 2:
                        _, concept = event
                        self._add(concept)

                    elif len(event) == 3:
                        _, concept, _ = event
                        self._add(concept)

        print(f"[VOCAB] Size = {len(self.token_to_id)}")
        return self

    def _add(self, token):
        if token is None:
            return
        if token not in self.token_to_id:
            self.token_to_id[token] = self.counter
            self.counter += 1

    def encode(self, token):
        return self.token_to_id.get(token, 0)