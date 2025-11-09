from typing import Sequence, TypeVar, List, Optional, Tuple

import secrets
import asyncio
import hashlib
import math
import itertools

T = TypeVar("T")

class TRandom:
    def __init__(self) -> None:
        self.__next_gauss: Optional[float] = None
    
    def _clear_internals(self) -> None:
        self.__next_gauss = None

    async def randint(self, a: int, b: int) -> int:
        if (a > b):
            raise ValueError("'a' Must Be <= 'b'")
        
        return await asyncio.to_thread(secrets.randbelow, b - a + 1) + a
    
    async def randrange(self, start: int, stop: Optional[int] = None, step: int = 1) -> int:
        if (step == 0):
            raise ValueError("Step Must Not Be '0'")

        if (stop is None):
            start, stop = 0, start

        width: int = stop - start
        
        if (step > 0 and width <= 0) or (step < 0 and width >= 0):
            raise ValueError("Empty Range For 'randrange()' With Given Step")

        n: int = (abs(width) + abs(step) - 1) // abs(step)
        r: int = await asyncio.to_thread(secrets.randbelow, n)

        return start + step * r
    
    async def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        if (self.__next_gauss is not None):
            z: float = self.__next_gauss
            self.__next_gauss = None
        else:
            z0, z1 = await self._p_gauss()
            self.__next_gauss = z1
            z = z0

        return mu + sigma * z
    
    async def _p_gauss(self) -> Tuple[float, float]:
        u1: float = await self.uniform(1e-12, 1.0)
        u2: float = await self.uniform(0.0, 1.0)

        r: float = math.sqrt(-2.0 * math.log(u1))
        theta: float = 2.0 * math.pi * u2

        z0: float = r * math.cos(theta)
        z1: float = r * math.sin(theta)

        return z0, z1
    
    async def random(self) -> float:
        bits: int = await asyncio.to_thread(secrets.randbits, 53)

        return (bits + 0.5) / (1 << 53)
    
    async def mix(self, *args: bytes, secure: bool = True) -> bytes:
        m: hashlib._Hash = hashlib.sha256()
        total_salt_size: int = len(args) * 16 if (secure) else 0
        salt_bytes: bytes = await asyncio.to_thread(secrets.token_bytes, total_salt_size) if (secure) else b""
        salt_offset: int = 0

        for a in args:
            m.update(a)
            
            if (secure):
                m.update(salt_bytes[salt_offset:salt_offset + 16])
                salt_offset += 16
        
        return m.digest()

    async def choice(self, seq: Sequence[T]) -> T:
        if (not seq):
            raise IndexError("Cannot Choose From An Empty Sequence")

        r: int = await asyncio.to_thread(secrets.randbelow, len(seq))
        return seq[r]
    
    async def w_choice(self, seq: Sequence[T], weights: Sequence[float]) -> T:
        if (len(seq) != len(weights)):
            raise ValueError("Sequence And Weight Lengths Must Match")
        
        if (any(w < 0 for w in weights)):
            raise ValueError("Weights Cannot Be Negative")
        
        total: float = sum(weights)

        if (total <= 0):
            raise ValueError("Total Weight Must Be Positive")
        
        normalized: List[float] = [w / total for w in weights]
        cumulative: List[float] = list(itertools.accumulate(normalized))
        r: float = await self.uniform(0.0, 1.0)

        for i, cw in enumerate(cumulative):
            if (r <= cw):
                return seq[i]
        
        return seq[-1]
    
    async def w_choices(self, seq: Sequence[T], weights: Sequence[float], k: int) -> List[T]:
        if (len(seq) != len(weights)):
            raise ValueError("Sequence And Weight Lengths Must Match")
        
        if (any(w < 0 for w in weights)):
            raise ValueError("Weights Cannot Be Negative")
        
        total: float = sum(weights)

        if (total <= 0):
            raise ValueError("Total Weight Must Be Positive")
        
        normalized: List[float] = [w / total for w in weights]
        cumulative: List[float] = list(itertools.accumulate(normalized))
        result: List[T] = []

        for _ in range(k):
            r: float = await self.uniform(0.0, 1.0)

            for i, cw in enumerate(cumulative):
                if (r <= cw):
                    result.append(seq[i])
                    break
            else:
                result.append(seq[-1])
        
        return result

    async def shuffle(self, lst: List[T]) -> None:
        n: int = len(lst)

        if (n <= 1):
            return
        
        random_indices: List[int] = await asyncio.to_thread(
            lambda: [secrets.randbelow(i + 1) for i in range(1, n)]
        )

        for i, j in enumerate(random_indices, start=1):
            idx: int = n - i
            lst[idx], lst[j] = lst[j], lst[idx]
    
    async def uniform(self, a: float, b: float) -> float:
        r: float = await self.random()

        return a + (b - a) * r
    
    async def sample(self, seq: Sequence[T], k: int) -> List[T]:
        if (k > len(seq)):
            raise ValueError("Sample Larger Than Population")

        if (k == 0):
            return []
        
        lst: List[T] = list(seq)
        n: int = len(lst)

        random_indices: List[int] = await asyncio.to_thread(
            lambda: [secrets.randbelow(n - i) for i in range(k)]
        )

        for i in range(k):
            j: int = i + random_indices[i]
            lst[i], lst[j] = lst[j], lst[i]
        
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
