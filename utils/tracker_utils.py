import heapq
import random

class HardestExamplesTracker:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.heap = []
        self.total_added = 0

    def add(self, loss, example, label):
        # Ensure the label is either "virtual" or "gt"
        # assert label in ["virtual", "gt"], "Label must be 'virtual' or 'gt'"
        
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (loss, example, label))
            self.total_added += 1
        elif loss > self.heap[0][0]:
            heapq.heappushpop(self.heap, (loss, example, label))

    def get_hardest_examples(self):
        # Sort by loss and return examples with their labels
        return [(example, label) for loss, example, label in sorted(self.heap, reverse=True)]

    def get_random_example(self):
        if not self.heap:
            return None
        _, example, label = random.choice(self.heap)
        return example, label

    def get_hardest_example(self):
        if not self.heap:
            return None
        _, example, label = max(self.heap, key=lambda x: x[0])
        return example, label

    def get_size(self):
        return self.total_added


