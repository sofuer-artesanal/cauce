from collections import deque
from typing import List
from .models import ChatMessage
from .config import MAX_WINDOW_SIZE


class ShortTermBuffer:
    def __init__(self) -> None:
        self._dq = deque()

    def append(self, message: ChatMessage) -> None:
        self._dq.append(message)

    def is_full(self) -> bool:
        return len(self._dq) >= MAX_WINDOW_SIZE

    def pop_oldest_batch(self, batch_size: int) -> List[ChatMessage]:
        out = []
        for _ in range(min(batch_size, len(self._dq))):
            out.append(self._dq.popleft())
        return out

    def snapshot(self) -> List[ChatMessage]:
        return list(self._dq)

    def to_list(self) -> List[ChatMessage]:
        return self.snapshot()

    def replace_all(self, messages: List[ChatMessage]) -> None:
        self._dq = deque(messages)

    def size(self) -> int:
        return len(self._dq)
