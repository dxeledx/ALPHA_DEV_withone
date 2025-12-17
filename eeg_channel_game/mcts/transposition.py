from __future__ import annotations

from eeg_channel_game.mcts.node import Node


class TranspositionTable:
    def __init__(self):
        self.table: dict[int, Node] = {}

    def get(self, key: int) -> Node | None:
        return self.table.get(int(key))

    def put(self, key: int, node: Node) -> None:
        self.table[int(key)] = node

    def clear(self) -> None:
        self.table.clear()

