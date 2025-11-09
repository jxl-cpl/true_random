from typing import Sequence, TypeVar, List, Optional

import secrets
import asyncio
import hashlib
import math

T = TypeVar("T")

class TRandom:
    def __init__(self):
        self._next_gauss: float | None = None

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
    
    async def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        if (self._next_gauss is not None):
            z = self._next_gauss
            self._next_gauss = None

            return mu + z * sigma
        
        z0, z1 = await self.p_gauss()
        self._next_gauss = z1

        return mu + z0 * sigma
    
    async def p_gauss(self, mu: float = 0.0, sigma: float = 1.0) -> tuple[float, float]:
        u1 = max(await self.random(), 1e-15)
        u2 = await self.random()
        r = (-2.0 * math.log(u1)) ** 0.5
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)

        return z0, z1
    
    async def random(self) -> float:
        bits = await asyncio.to_thread(secrets.randbits, 53)

        return (bits + 0.5) / (1 << 53)

    async def choice(self, seq: Sequence[T]) -> T:
        if (not seq):
            raise IndexError("Cannot Choose From An Empty Sequence")

        r = await asyncio.to_thread(secrets.randbelow, len(seq))
        return seq[r]
    
    async def w_choice(self, seq: Sequence[T], weights: Sequence[float]) -> T:
        if (not seq or len(seq) != len(weights)):
            raise ValueError("Sequences Must Have Same Length And Not Be Empty")
        
        total = sum(weights)
        r = (await self.random()) * total

        for item, w in zip(seq, weights):
            r -= w

            if (r <= 0):
                return item
        
        return seq[-1]
    
    async def shuffle(self, lst: List[T]) -> None:
        for i in reversed(range(1, len(lst))):
            # j = await asyncio.to_thread(secrets.randbelow, i + 1)
            j = secrets.randbelow(i + 1)

            lst[i], lst[j] = lst[j], lst[i]
    
    async def uniform(self, a: float, b: float) -> float:
        r = await self.random()

        return a + (b - a) * r
    
    async def sample(self, seq: Sequence[T], k: int) -> List[T]:
        if (k > len(seq)):
            raise ValueError("Sample Larger Than Population")

        lst = list(seq)

        await self.shuffle(lst)

        return lst[:k]

    """
    USED FOR ANOTHER PROJECT, PLEASE IGNORE.
    """
    async def d_shuffle(self, indices: List[int], password: str) -> None:
        seed = int(hashlib.sha256(password.encode()).hexdigest(), 16)

        for i in range(len(indices) - 1, 0, -1):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            j = seed % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
