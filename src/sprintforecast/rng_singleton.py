from numpy.random import Generator, default_rng

import threading

class RNGSingleton:
    _lock = threading.Lock()
    _rng: Generator | None = None

    @classmethod
    def rng(cls) -> Generator:
        if cls._rng is None:
            with cls._lock:
                if cls._rng is None:
                    cls._rng = default_rng()
        return cls._rng