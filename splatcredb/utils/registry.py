from collections.abc import Callable

class BaselineRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Callable] = {}
    def register(self, name: str, fn: Callable) -> None:
        self._items[name]=fn
    def get(self, name: str) -> Callable:
        if name not in self._items:
            raise KeyError(f"Unknown baseline: {name}")
        return self._items[name]
