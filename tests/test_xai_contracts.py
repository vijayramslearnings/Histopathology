from __future__ import annotations

import numpy as np
import pytest

from backend.xai.attention_maps import attention_heatmap


def test_attention_heatmap_normalizes_scale_patch_values() -> None:
    values = np.arange(48, dtype=np.float32)
    heatmap = attention_heatmap(values, scales=3, grid_size=4)
    assert heatmap.shape == (4, 4)
    assert float(heatmap.min()) == 0.0
    assert float(heatmap.max()) == 1.0


def test_attention_heatmap_rejects_wrong_token_count() -> None:
    with pytest.raises(ValueError, match="Expected"):
        attention_heatmap([0.5] * 5, scales=3, grid_size=4)