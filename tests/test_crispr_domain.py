# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

import math

import pytest

from ailee.domains.CRISPR import AileeCRISPRTrustLayer


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf, -1.0, 101.0, True, "85"])
def test_crispr_rejects_invalid_policy_thresholds(threshold):
    with pytest.raises(ValueError):
        AileeCRISPRTrustLayer(threshold)


@pytest.mark.parametrize(
    ("grna", "target"),
    [
        (None, "ATCGATCGATCGATCGATCGAGG"),
        ("ATCGATCGATCGATCGATCG", None),
        ("ATCGATCGATCGATCGATXG", "ATCGATCGATCGATCGATCGAGG"),
        ("ATCGATCGATCGATCGATCG", "ATCGATCGATCGATCGATC?AGG"),
        ("", "AGG"),
    ],
)
def test_crispr_malformed_sequences_fail_closed(grna, target):
    decision = AileeCRISPRTrustLayer().evaluate_sequence(grna, target)
    assert decision["status"] == "REJECTED"
    assert decision["is_safe_to_execute"] is False
    assert decision["trust_score"] == 0.0

