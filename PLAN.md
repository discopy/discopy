# Plan: Fix Issue #307 — `str(Ty(""))` incorrectly returns `'Ty()'`

## Issue Summary

`Ty("")` creates a monoidal type with a single `Ob` with an empty-string name. When converted to a string, it is indistinguishable from `Ty()` (the empty/unit type):

```python
str(Ty())    # 'Ty()'  -- correct
str(Ty(""))  # 'Ty()'  -- WRONG, should be 'Ty("")'
```

## Root Cause

Two cooperating causes:

1. **`Ob.__str__`** in `discopy/cat.py` line 118: `str(Ob(""))` returns `""` (empty string), which is falsy.

2. **`Ty.__str__`** in `discopy/monoidal.py` line 165:
   ```python
   def __str__(self):
       return ' @ '.join(map(str, self.inside)) or type(self).__name__ + '()'
   ```
   `' @ '.join([""])` produces `""` (falsy), so the `or` short-circuits to `"Ty()"`.

## Files to Change

| File | Change |
|---|---|
| `discopy/monoidal.py` | Fix `Ty.__str__` — check `self.inside` emptiness, not string falsiness |
| `test/syntax/monoidal.py` | Add regression tests for `str(Ty(""))` |

## Implementation Steps

### Step 1: Fix `Ty.__str__` in `discopy/monoidal.py`

**Current code** (line 165):
```python
def __str__(self):
    return ' @ '.join(map(str, self.inside)) or type(self).__name__ + '()'
```

**New code:**
```python
def __str__(self):
    if not self.inside:
        return type(self).__name__ + '()'
    parts = []
    for ob in self.inside:
        s = str(ob)
        parts.append('"{}"'.format(s) if s == '' else s)
    return ' @ '.join(parts)
```

This correctly produces:
- `str(Ty())` → `'Ty()'`
- `str(Ty(""))` → `'Ty("")'`
- `str(Ty("x"))` → `'x'`  (unchanged)
- `str(Ty("x", ""))` → `'x @ ""'`

### Step 2: Add regression tests in `test/syntax/monoidal.py`

Extend `test_Ty_str` to include:
```python
assert str(Ty("")) == 'Ty("")'
assert str(Ty()) != str(Ty(""))
assert str(Ty("x", "")) == 'x @ ""'
```

## Edge Cases

- **`Bubble.to_drawing()`** uses `Ty("")` as a spacer wire — not affected, since `to_drawing()` calls `str(ob)` per `Ob`, not `str(Ty(...))`.
- **`PRO.__str__`** has its own `__str__`, so it's unaffected.
- **Subclasses of `Ty`** inherit the fix automatically.
- **`Ty("")` vs `Ty()` equality** is already correctly handled via `__eq__`; this fix only improves the string representation.
- **`Ty.__init__`** calls `super().__init__(str(self))` to set `self.name`, so after the fix `Ty("").name` will be `'Ty("")'` instead of `'Ty()'` — more correct behavior.
