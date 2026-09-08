from __future__ import annotations

from collections.abc import Hashable, Iterable


class UnionFind:
    def __init__(self, items: Iterable[Hashable] = ()) -> None:
        self.parent: dict[Hashable, Hashable] = {}
        self.rank: dict[Hashable, int] = {}
        for item in items:
            self.add(item)

    def add(self, item: Hashable) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: Hashable) -> Hashable:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: Hashable, right: Hashable) -> Hashable:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return root_left
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1
        return root_left

    def groups(self) -> list[list[Hashable]]:
        grouped: dict[Hashable, list[Hashable]] = {}
        for item in self.parent:
            grouped.setdefault(self.find(item), []).append(item)
        return [sorted(items, key=str) for _, items in sorted(grouped.items(), key=lambda pair: str(pair[0]))]

