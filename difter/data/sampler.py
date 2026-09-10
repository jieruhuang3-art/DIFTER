from collections import defaultdict
from itertools import combinations
import math
from .collate import collate_flows


class DomainClassSupportSampler:
    """Build source-only support batches with two classes and two environments."""

    def __init__(self, flows, factor):
        cells = defaultdict(list)
        for flow in flows:
            cells[(flow["label"], flow["windows"][0]["environment"][factor])].append(flow)
        self.cells, self.tuples, self.cursor, self.cell_cursor = cells, [], 0, defaultdict(int)
        classes = sorted({key[0] for key in cells}); environments = sorted({key[1] for key in cells})
        for first, second in combinations(classes, 2):
            for env_a, env_b in combinations(environments, 2):
                if all(cells[(label, env)] for label in (first, second) for env in (env_a, env_b)) and all(
                    len(cells[(label, env_a)]) + len(cells[(label, env_b)]) >= 4 for label in (first, second)):
                    self.tuples.append((first, second, env_a, env_b))
        if not self.tuples:
            raise ValueError("no eligible class/environment support tuple")

    def next(self):
        first, second, env_a, env_b = self.tuples[self.cursor % len(self.tuples)]
        self.cursor += 1
        output = []
        for label in (first, second):
            count_a, count_b = len(self.cells[(label, env_a)]), len(self.cells[(label, env_b)])
            allocation = (2, 2) if count_a >= 2 and count_b >= 2 else ((3, 1) if count_a >= 3 else (1, 3))
            for env, count in ((env_a, allocation[0]), (env_b, allocation[1])):
                cell = (label, env); values = self.cells[cell]; start = self.cell_cursor[cell]
                output.extend(values[(start + index) % len(values)] for index in range(count))
                self.cell_cursor[cell] = (start + count) % len(values)
        return output


class MainFlowBatchIterator:
    """Deterministic class-balanced eight-flow main sampler."""

    def __init__(self, flows, max_windows=16, max_length=128):
        self.by_class = defaultdict(list)
        for flow in flows: self.by_class[flow["label"]].append(flow)
        self.classes = sorted(self.by_class)
        self.batches_per_epoch = math.ceil(len(flows) / 8)
        self.max_windows, self.max_length = max_windows, max_length

    def __iter__(self):
        successful = 0
        while True:
            epoch, batch_index = divmod(successful, self.batches_per_epoch)
            chosen = []
            for slot in range(8):
                label = self.classes[(batch_index * 8 + slot) % len(self.classes)]
                candidates = self.by_class[label]
                position = (epoch * self.batches_per_epoch * 8 + batch_index * 8 + slot) // len(self.classes)
                chosen.append(candidates[position % len(candidates)])
            yield collate_flows(chosen, self.max_windows, self.max_length, epoch, True)
            successful += 1


class SupportFlowBatchIterator:
    def __init__(self, flows, factor, max_windows=16, max_length=128):
        self.sampler = DomainClassSupportSampler(flows, factor)
        self.max_windows, self.max_length = max_windows, max_length

    def __iter__(self):
        epoch = 0
        while True:
            yield collate_flows(self.sampler.next(), self.max_windows, self.max_length, epoch, True)
            epoch += 1
