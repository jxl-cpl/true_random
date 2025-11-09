from typing import Sequence, TypeVar, List, Optional

import secrets
import asyncio
import hashlib

T = TypeVar("T")

class TRandom:
    async def randint(self, a: int, b: int) -> int:
        if (a > b):
            raise ValueError("'a' Must Be <= 'b'")
        
        return await asyncio.to_thread(secrets.randbelow, b - a + 1) + a
    
    async def randrange(self, start: int, stop: Optional[int] = None, step: int = 1) -> int:
        if (step == 0):
            raise ValueError("Step Must Not Be '0'")

        if (stop is None):
            start, stop = 0, start

        width = stop - start
        
        if (step > 0 and width <= 0) or (step < 0 and width >= 0):
            raise ValueError("Empty Range For 'randrange()' With Given Step")

        n = abs((width + step - (1 if (step) > 0 else -1)) // step)
        r = await asyncio.to_thread(secrets.randbelow, n)

        return start + step * r
    
    async def random(self) -> float:
        bits = await asyncio.to_thread(secrets.randbits, 53)

        return (bits + 0.5) / (1 << 53)
    
    async def choice(self, seq: Sequence[T]) -> T:
        if (not seq):
            raise IndexError("Cannot Choose From An Empty Sequence.")

        r = await asyncio.to_thread(secrets.randbelow, len(seq))
        return seq[r]
    
    async def shuffle(self, lst: List[T]) -> None:
        for i in reversed(range(1, len(lst))):
            j = await asyncio.to_thread(secrets.randbelow, i + 1)
            lst[i], lst[j] = lst[j], lst[i]

    """
    USED FOR ANOTHER PROJECT, PLEASE IGNORE.
    """
    async def d_shuffle(self, indices: List[int], password: str) -> None:
        seed = int(hashlib.sha256(password.encode()).hexdigest(), 16)

        for i in range(len(indices) - 1, 0, -1):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            j = seed % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
