import sympy as sp

class SymbolicMetrics:
    p = sp.symbols("p", positive=True)
    brier = p * (1 - p)

    @classmethod
    def brier_derivative(cls) -> sp.Expr:
        return sp.diff(cls.brier, cls.p)