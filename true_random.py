__version__ = "1.0.1"
__repo__ = "https://raw.githubusercontent.com/jxl-cpl/true_random/refs/heads/main/true_random.py"

from typing import Sequence, TypeVar, List, Optional, Tuple, Dict, Set
from pathlib import Path

import secrets
import asyncio
import hashlib
import math
import itertools
import bisect
import numpy
import string
import sys
import urllib.request

T = TypeVar("T")

class TRandom:
    __slots__ = ("__next_gauss", "__weight_cache")

    _two_pi: float = 2.0 * math.pi
    _inv_2_pow_53: float = 1.0 / (1 << 53)

    def __init__(self) -> None:
        self.__next_gauss: Optional[float] = None
        self.__weight_cache: Dict[Tuple[float, ...], List[float]] = {}
    
    def _clear_internals(self) -> None:
        self.__next_gauss = None
    
    def _get_cumulative(self, weights: Sequence[float]) -> List[float]:
        weights_tuple: Tuple[float, ...] = tuple(weights)

        if (weights_tuple in self.__weight_cache):
            return self.__weight_cache[weights_tuple]
        
        total: float = sum(weights)

        if (total <= 0):
            raise ValueError("Total Weight Must Be Positive")

        normalized: List[float] = [w / total for w in weights]
        cumulative: List[float] = list(itertools.accumulate(normalized))
        self.__weight_cache[weights_tuple] = cumulative

        return cumulative

    async def randint(self, a: int, b: int) -> int:
        if (a > b):
            raise ValueError("'a' Must Be <= 'b'")
        
        return secrets.randbelow(b - a + 1) + a
    
    async def randints(self, a: int, b: int, k: int) -> List[int]:
        if (a > b):
            raise ValueError("'a' Must Be <= 'b'")
        
        range_size: int = b - a + 1

        return await asyncio.to_thread(
            lambda: [secrets.randbelow(range_size) + a for _ in range(k)]
        )
    
    async def randrange(self, start: int, stop: Optional[int] = None, step: int = 1) -> int:
        if (step == 0):
            raise ValueError("Step Must Not Be '0'")

        if (stop is None):
            start, stop = 0, start

        width: int = stop - start
        
        if (step > 0 and width <= 0) or (step < 0 and width >= 0):
            raise ValueError("Empty Range For 'randrange()' With Given Step")

        n: int = (abs(width) + abs(step) - 1) // abs(step)
        r: int = secrets.randbelow(n)

        return start + step * r
    
    async def randfloats(self, k: int) -> List[float]:
        bits_list: List[int] = await asyncio.to_thread(
            lambda: [secrets.randbits(53) for _ in range(k)]
        )

        return [(bits + 0.5) / (1 << 53) for bits in bits_list]
    
    async def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        if (self.__next_gauss is not None):
            z: float = self.__next_gauss
            self.__next_gauss = None
        else:
            z0, z1 = await self._p_gauss()
            self.__next_gauss = z1
            z = z0

        return mu + sigma * z
    
    async def gausses(self, mu: float = 0.0, sigma: float = 1.0, k: int = 1) -> List[float]:
        result: List[float] = []
        pairs_needed: int = (k + 1) // 2

        for _ in range(pairs_needed):
            z0, z1 = await self._p_gauss()
            result.append(mu + sigma * z0)

            if (len(result) < k):
                result.append(mu + sigma * z1)
        
        return result[:k]
    
    async def _p_gauss(self) -> Tuple[float, float]:
        u1: float = await self.uniform(1e-12, 1.0)
        u2: float = await self.uniform(0.0, 1.0)

        r: float = math.sqrt(-2.0 * math.log(u1))
        theta: float = self._two_pi * u2

        z0: float = r * math.cos(theta)
        z1: float = r * math.sin(theta)

        return z0, z1
    
    async def random(self) -> float:
        bits: int = secrets.randbits(53)
        return (bits + 0.5) * self._inv_2_pow_53
    
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

        r: int = secrets.randbelow(len(seq))
        return seq[r]
    
    async def w_choice(self, seq: Sequence[T], weights: Sequence[float]) -> T:
        if (len(seq) != len(weights)):
            raise ValueError("Sequence And Weight Lengths Must Match")
        
        if (any(w < 0 for w in weights)):
            raise ValueError("Weights Cannot Be Negative")
        
        cumulative: List[float] = self._get_cumulative(weights)
        r: float = await self.uniform(0.0, 1.0)
        idx: int = bisect.bisect_right(cumulative, r)

        return seq[min(idx, len(seq) - 1)]
    
    async def w_choices(self, seq: Sequence[T], weights: Sequence[float], k: int) -> List[T]:
        if (len(seq) != len(weights)):
            raise ValueError("Sequence And Weight Lengths Must Match")
        
        if (any(w < 0 for w in weights)):
            raise ValueError("Weights Cannot Be Negative")

        cumulative: List[float] = self._get_cumulative(weights)
        result: List[T] = []

        for _ in range(k):
            r: float = await self.uniform(0.0, 1.0)
            idx: int = bisect.bisect_right(cumulative, r)
            result.append(seq[min(idx, len(seq) - 1)])
        
        return result

    async def shuffle(self, lst: List[T]) -> None:
        n: int = len(lst)

        if (n <= 1):
            return
        
        random_indices: List[int] = await asyncio.to_thread(
            lambda: [secrets.randbelow(i + 1) for i in range(1, n)]
        )

        for i in range(len(random_indices)):
            idx: int = n - i - 1
            j: int = random_indices[i]
            lst[idx], lst[j] = lst[j], lst[idx]
    
    async def uniform(self, a: float, b: float) -> float:
        r: float = await self.random()

        return a + (b - a) * r

    async def uniforms(self, a: float, b: float, k: int) -> List[float]:
        if (k > 100):
            bits_array: numpy.ndarray = await asyncio.to_thread(lambda: numpy.array([secrets.randbits(53) for _ in range(k)], dtype=numpy.float64))
            _randoms: numpy.ndarray = (bits_array + 0.5) / (1 << 53)

            return ((a + (b - a) * _randoms).tolist())
        
        randoms: List[float] = await self.randfloats(k)
        return [a + (b - a) * r for r in randoms]
    
    async def sample(self, seq: Sequence[T], k: int) -> List[T]:
        if (k > len(seq)):
            raise ValueError("Sample Larger Than Population")

        if (k == 0):
            return []

        n: int = len(seq)

        if (k == 1):
            return [await self.choice(seq)]
        
        if (k * 10 < n):
            seen: Set[int] = set()
            result: List[T] = []

            while (len(result) < k):
                idx: int = secrets.randbelow(n)

                if (idx not in seen):
                    seen.add(idx)
                    result.append(seq[idx])
            
            return result
        
        lst: List[T] = list(seq)
        random_indices: List[int] = await asyncio.to_thread(
            lambda: [secrets.randbelow(n - i) for i in range(k)]
        )

        for i in range(k):
            j: int = i + random_indices[i]
            lst[i], lst[j] = lst[j], lst[i]
        
        return lst[:k]
    
    async def token_bytes(self, n: int) -> bytes:
        if (n < 0):
            raise ValueError("Byte Count Must Be Non-Negative")
        
        return await asyncio.to_thread(secrets.token_bytes, n)
    
    async def token_hex(self, n: int) -> str:
        if (n < 0):
            raise ValueError("Length Must Be Non-Negative")
        
        return await asyncio.to_thread(secrets.token_hex, n)

    async def gamma(self, alpha: float, beta_param: float = 1.0) -> float:
        if (alpha <= 0 or beta_param <= 0):
            raise ValueError("Alpha And Beta Must Be Positive")
        
        if (alpha < 1):
            u: float = await self.uniform(0.0, 1.0)

            return await self.gamma(alpha + 1.0, beta_param) * (u ** (1.0 / alpha))
        
        d: float = alpha - 1.0 / 3.0
        c: float = 1.0 / math.sqrt(9.0 * d)

        while (True):
            z: float = await self.gauss()
            v: float = (1.0 + c * z) ** 3

            if (v <= 0):
                continue

            u: float = await self.uniform(0.0, 1.0)

            if (u < 1.0 - 0.0331 * (z ** 4)):
                return d * v / beta_param
            
            if (math.log(u) < 0.5 * (z ** 2) + d * (1.0 - v + math.log(v))):
                return d * v / beta_param

    async def beta(self, alpha: float, beta_param: float) -> float:
        if (alpha <= 0 or beta_param <= 0):
            raise ValueError("Alpha And Beta Must Be Positive")
        
        x: float = await self.gamma(alpha, 1.0)
        y: float = await self.gamma(beta_param, 1.0)

        return x / (x + y)
    
    async def exponential(self, lambd: float = 1.0) -> float:
        if (lambd <= 0):
            raise ValueError("Lambda Must Be Positive")
        
        u: float = await self.uniform(1e-12, 1.0)

        return -math.log(u) / lambd

    async def coin_flip(self) -> bool:
        return secrets.randbits(1) == 1
    
    async def r_string(self, length: int, alphabet: Optional[str] = None) -> str:
        if (length < 0):
            raise ValueError("Length Must Be Non-Negative")
        
        if (alphabet is None):
            alphabet = string.ascii_letters + string.digits
        
        n = len(alphabet)
        indices: List[int] = await asyncio.to_thread(
            lambda: [secrets.randbelow(n) for _ in range(length)]
        )

        return "".join(alphabet[i] for i in indices)

async def update() -> None:
    try:
        def _fetch_remote() -> str:
            with urllib.request.urlopen(__repo__) as response:
                if (response.status != 200):
                    raise RuntimeError(f"[!] Failed To Fetch Update | {response.status}.")
                
                return response.read().decode("utf-8")
        
        data: str = await asyncio.to_thread(_fetch_remote)
        remote_version: Optional[str] = None

        for line in data.splitlines():
            if (line.startswith("__version__")):
                remote_version = line.split("=")[1].strip().strip('"\'')
                break
        
        if (remote_version is None):
            print("[!] Couldn't Find Version Information.")
            return
        
        if (remote_version == __version__):
            print(f"[?] You Are Already Using Latest Version ({__version__}).")
            return
        
        path: Path = Path(sys.argv[0]).resolve()

        def _write_file() -> None:
            path.write_text(data, encoding="utf-8")
        
        await asyncio.to_thread(_write_file)

        print(f"[+] Updated To Latest Version '{remote_version}'.")

    except Exception as error:
        print(f"[!] Update Failed | {error}.")

if (__name__ == "__main__"):
    if ("-update" in sys.argv):
        asyncio.run(update())
        sys.exit(0)
