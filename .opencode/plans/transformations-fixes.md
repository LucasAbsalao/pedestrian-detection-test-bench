# Implementation Plan: Fix transformations.py Issues

## Issues to Fix

### Issue 2: Dynamic Channel Handling in `fog()` (Line 16)
**Current**: Hardcoded `repeat(..., 3, axis=2)` assumes 3-channel RGB
**Fix**: Detect channels from image shape, repeat dynamically

```python
# Replace lines 15-18:
if len(depth_map.shape) < 3:
    channels = image.shape[2] if image.ndim == 3 else 1
    beer_lambert = np.repeat(np.exp((-k)*depth_map)[..., np.newaxis], channels, axis=2)
else:
    beer_lambert = depth_map
```

### Issue 5: Atmospheric Light - Add `per_channel_airlight` Parameter
**Decision**: Keep current behavior as default, add optional parameter

```python
# Update function signature line 5:
def fog(image, depth_map, minimum_distance, airlight=None, per_channel_airlight=False):

# Replace lines 10-13:
if airlight is None:
    if per_channel_airlight and image.ndim == 3:
        atmospheric_light = np.mean(image, axis=(0, 1))
    else:
        atmospheric_light = np.mean(image)
else:
    atmospheric_light = airlight
```

### Issue 6: Add Type Hints to All Functions
**Changes**: Add imports + type hints to `fog`, `magnitude_of_gradient`, `minimax_normalization`

```python
# Top of file (after imports):
from __future__ import annotations
from numpy.typing import NDArray
from typing import Optional

# Function signatures:
def fog(image: NDArray, depth_map: NDArray, minimum_distance: float,
        airlight: Optional[float] = None, per_channel_airlight: bool = False) -> NDArray:

def magnitude_of_gradient(image: NDArray) -> NDArray:

def minimax_normalization(image: NDArray, dtype: type = np.uint8) -> NDArray:
```

### Bonus Fix: Issue 3 - Add `np.clip()` in `gaussian_noise()`
**Current**: Line 26 `new_image = image + noise` can produce invalid values
**Fix**: 
```python
new_image = np.clip(image + noise, 0, 255).astype(image.dtype)
```

---

## Files to Modify
- `/home/lucas/Documents/computer_vision/src/computer_vision/utils/transformations.py`

## Testing
After implementation, verify:
1. `fog()` works with 1, 3, 4 channel images
2. `per_channel_airlight=True` produces per-channel atmospheric light
3. `gaussian_noise()` output stays in valid range
4. All type hints pass static analysis (mypy/pyright)
5. Existing functionality unchanged functions still work