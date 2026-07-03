#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests del núcleo D2 (curva de resolución).

Run:  /Users/hectorbecerrilvillamil/miniconda3/bin/python augmented/tests_resolution.py
Exit code 0 = PASS, 1 = FAIL.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented.core import bin_of


def test_bin_of_truncation():
    # cap None => conteo completo (identidad)
    assert bin_of(0, None) == 0
    assert bin_of(3, None) == 3
    # cap = 1 => binario: 0 vs >=1
    assert bin_of(0, 1) == 0
    assert bin_of(1, 1) == 1
    assert bin_of(5, 1) == 1
    # cap = 2 => {0, 1, >=2}
    assert [bin_of(r, 2) for r in range(5)] == [0, 1, 2, 2, 2]
    # aísla {0}: ningún r>=1 cae en el bin 0
    for cap in (1, 2, 3):
        assert bin_of(0, cap) == 0
        assert all(bin_of(r, cap) >= 1 for r in range(1, 8))


# ---- Test runner ----
if __name__ == "__main__":
    test_fns = sorted(
        [(name, obj) for name, obj in globals().items()
         if name.startswith("test_") and callable(obj)],
        key=lambda x: x[0],
    )
    passed = failed = 0
    for name, fn in test_fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    if failed:
        sys.exit(1)
