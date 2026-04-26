class AileeCRISPRTrustLayer:
    """
    A standalone, production-ready Python module acting as a strict safety
    and verification filter for genetic sequences (specifically CRISPR gRNA)
    before they are passed to downstream computational simulations.
    """

    # Mock database of hazardous sequences for Gate 3.1
    # Sequences here are representative substrings that might trigger a hazard alert.
    HAZARDOUS_SEQUENCES = {
        "GATTACAGATTACA",
        "TGCATGCTGCATGC",
        "CCCCCCGGGGGG"
    }

    def __init__(self, trust_threshold: float = 85.0):
        """
        Initialize the trust layer.

        Args:
            trust_threshold (float): The minimum trust score required to pass
                                     the Grace Layer (distal tolerance check).
        """
        self.trust_threshold = trust_threshold

    def evaluate_sequence(self, grna: str, target_dna: str) -> dict:
        """
        Evaluate a gRNA sequence against a target DNA sequence using a sequential gating process.

        Gate 3.1: External Prior Art Sync (Hazard check)
        Threshold Validation: PAM Check (NGG at 3' end)
        Threshold Validation: Seed Region (10 bases adjacent to PAM must match exactly)
        Gate 3.2: Grace Layer - Distal Tolerance (Weighted penalty for distal mismatches)

        Args:
            grna (str): The proposed gRNA sequence (typically 20 bases).
            target_dna (str): The target DNA sequence including the PAM (e.g., 20 bases + 3 PAM = 23 bases).

        Returns:
            dict: The standardized output schema detailing the status and trust score.
        """
        grna = grna.upper()
        target_dna = target_dna.upper()

        # Gate 3.1 (External Prior Art Sync)
        for haz_seq in self.HAZARDOUS_SEQUENCES:
            if haz_seq in grna:
                return {
                    "status": "REJECTED",
                    "is_safe_to_execute": False,
                    "trust_score": 0.0,
                    "consensus_route": "Gate 3.1 -> Outright Rejected",
                    "sequence_analyzed": grna,
                    "log": f"Failure at Gate 3.1: Sequence contains hazardous motif '{haz_seq}'."
                }

        # Structure validation (basic lengths)
        if len(target_dna) < 3:
             return {
                "status": "REJECTED",
                "is_safe_to_execute": False,
                "trust_score": 0.0,
                "consensus_route": "Threshold Validation -> Outright Rejected",
                "sequence_analyzed": grna,
                "log": "Failure: Target DNA is too short to contain a PAM sequence."
            }

        if len(grna) < 10:
             return {
                "status": "REJECTED",
                "is_safe_to_execute": False,
                "trust_score": 0.0,
                "consensus_route": "Threshold Validation -> Outright Rejected",
                "sequence_analyzed": grna,
                "log": "Failure: gRNA is too short to contain a seed region."
            }

        # Threshold Validation (PAM Check)
        # NGG at 3' end of target DNA. N is any nucleotide (A, C, G, T).
        pam = target_dna[-3:]
        if not (pam[1] == 'G' and pam[2] == 'G'):
            return {
                "status": "REJECTED",
                "is_safe_to_execute": False,
                "trust_score": 0.0,
                "consensus_route": "Threshold Validation -> Outright Rejected",
                "sequence_analyzed": grna,
                "log": f"Failure at PAM Check: Target DNA lacks 'NGG' PAM at 3' end (found '{pam}')."
            }

        # Threshold Validation (Seed Region)
        # 10 bases on the gRNA immediately adjacent to the PAM (3' end of gRNA)
        # and 10 bases on the target DNA immediately 5' of the PAM.

        # We assume the gRNA aligns with the target DNA exactly preceding the PAM.
        # e.g., target_dna = [20 bases] + [3 PAM]
        # gRNA = [20 bases]
        if len(target_dna) - 3 != len(grna):
            return {
                "status": "REJECTED",
                "is_safe_to_execute": False,
                "trust_score": 0.0,
                "consensus_route": "Threshold Validation -> Outright Rejected",
                "sequence_analyzed": grna,
                "log": f"Failure: Length mismatch between gRNA ({len(grna)}) and Target DNA minus PAM ({len(target_dna) - 3})."
            }

        target_sequence = target_dna[:-3]
        grna_seed = grna[-10:]
        target_seed = target_sequence[-10:]

        if grna_seed != target_seed:
            return {
                "status": "REJECTED",
                "is_safe_to_execute": False,
                "trust_score": 0.0,
                "consensus_route": "Threshold Validation -> Outright Rejected",
                "sequence_analyzed": grna,
                "log": f"Failure at Seed Region Check: Mismatch in 10-base seed region. gRNA: '{grna_seed}', Target: '{target_seed}'."
            }

        # Gate 3.2 (Grace Layer - Distal Tolerance)
        # Remaining distal bases (furthest from PAM, i.e., 5' end of gRNA)
        grna_distal = grna[:-10]
        target_distal = target_sequence[:-10]

        trust_score = 100.0
        mismatches = 0
        for i in range(len(grna_distal)):
            if grna_distal[i] != target_distal[i]:
                trust_score -= 5.0
                mismatches += 1

        if trust_score < self.trust_threshold:
            return {
                "status": "REJECTED",
                "is_safe_to_execute": False,
                "trust_score": trust_score,
                "consensus_route": "Grace Layer -> Outright Rejected",
                "sequence_analyzed": grna,
                "log": f"Failure at Grace Layer: Trust score ({trust_score}) dropped below threshold ({self.trust_threshold}) due to {mismatches} distal mismatches."
            }

        # Consensus Passed
        return {
            "status": "ACCEPTED",
            "is_safe_to_execute": True,
            "trust_score": trust_score,
            "consensus_route": "Grace Layer -> Cleared",
            "sequence_analyzed": grna,
            "log": f"Sequence accepted. Seed match confirmed. Distal mismatches: {mismatches}. Final trust score: {trust_score}."
        }

if __name__ == "__main__":
    trust_layer = AileeCRISPRTrustLayer()

    print("Test 1: Perfect Match")
    # 20 base gRNA, 23 base target DNA
    grna1 =   "ATCGATCGATCGATCGATCG"
    target1 = "ATCGATCGATCGATCGATCGAGG"
    result1 = trust_layer.evaluate_sequence(grna1, target1)
    print(result1)
    assert result1["status"] == "ACCEPTED"
    assert result1["trust_score"] == 100.0

    print("\nTest 2: Fatal Seed Mismatch")
    grna2 =   "ATCGATCGATCGATCGATCA" # Last base changed from G to A
    target2 = "ATCGATCGATCGATCGATCGAGG"
    result2 = trust_layer.evaluate_sequence(grna2, target2)
    print(result2)
    assert result2["status"] == "REJECTED"
    assert "Seed Region Check" in result2["log"]

    print("\nTest 3: Grace Layer Distal Tolerance Pass")
    grna3 =   "TTCGATCGATCGATCGATCG" # First base changed from A to T (distal mismatch)
    target3 = "ATCGATCGATCGATCGATCGAGG"
    result3 = trust_layer.evaluate_sequence(grna3, target3)
    print(result3)
    assert result3["status"] == "ACCEPTED"
    assert result3["trust_score"] == 95.0

    print("\nTest 4: Grace Layer Distal Tolerance Fail")
    grna4 =   "TTTTTTCGATCGATCGATCG" # 4 distal mismatches (4 * -5 = -20) -> Score 80.0
    target4 = "ATCGATCGATCGATCGATCGAGG"
    result4 = trust_layer.evaluate_sequence(grna4, target4)
    print(result4)
    assert result4["status"] == "REJECTED"
    assert result4["trust_score"] == 80.0
    assert "Grace Layer" in result4["log"]

    print("\nTest 5: Fatal PAM Check")
    grna5 =   "ATCGATCGATCGATCGATCG"
    target5 = "ATCGATCGATCGATCGATCGAGA" # PAM changed from AGG to AGA
    result5 = trust_layer.evaluate_sequence(grna5, target5)
    print(result5)
    assert result5["status"] == "REJECTED"
    assert "PAM Check" in result5["log"]

    print("\nTest 6: Hazardous Sequence Fail")
    grna6 =   "GATTACAGATTACAGATCG" # Contains hazardous sequence
    target6 = "GATTACAGATTACAGATCGAGG"
    result6 = trust_layer.evaluate_sequence(grna6, target6)
    print(result6)
    assert result6["status"] == "REJECTED"
    assert "Gate 3.1" in result6["log"]

    print("\nAll tests passed successfully.")
