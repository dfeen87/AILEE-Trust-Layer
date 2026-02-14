"""
FEEN Confidence Scorer Bridge
==============================

This module provides a minimal adapter that allows AILEE to consume
confidence signals from FEEN hardware when available.

Architecture:
    AILEE consumes FEEN signals.
    FEEN provides physics primitives.
    This bridge translates between them.

No physics. No thresholds. No policy.
AILEE remains the source of truth for trust decisions.

Usage:
    from ailee.backends.feen.confidence_scorer import FEENConfidenceScorer
    
    scorer = FEENConfidenceScorer()
    
    if scorer.is_available():
        result = scorer.compute(raw_value, peers, history)
        # Returns: ConfidenceResult with score, stability, agreement, likelihood
    else:
        # Fall back to software implementation
        result = software_confidence_scorer.compute(...)
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    """
    Confidence computation result.
    Matches AILEE's internal ConfidenceResult structure.
    """
    score: float        # Weighted confidence score [0.0, 1.0]
    stability: float    # Temporal stability component [0.0, 1.0]
    agreement: float    # Peer agreement component [0.0, 1.0]
    likelihood: float   # Historical plausibility component [0.0, 1.0]


class FEENConfidenceScorer:
    """
    Bridge between AILEE and FEEN confidence computation.
    
    This class:
    - Detects FEEN availability
    - Translates AILEE inputs to FEEN primitives
    - Returns AILEE-compatible results
    - Falls back gracefully if FEEN unavailable
    
    Does NOT:
    - Implement physics
    - Make policy decisions
    - Override AILEE thresholds
    - Interpret results
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize FEEN bridge.
        
        Args:
            config: Optional FEEN hardware configuration.
                   If None, uses FEEN defaults.
        """
        self._feen_available = False
        self._pyfeen = None
        self._scorer = None
        
        # Attempt FEEN import
        try:
            import pyfeen.ailee as feen_ailee
            self._pyfeen = feen_ailee
            self._feen_available = True
            
            # Initialize FEEN hardware scorer
            if config is None:
                config = self._default_config()
            
            self._scorer = feen_ailee.PhononicConfidenceScorer(config)
            logger.info("FEEN confidence scorer initialized successfully")
            
        except ImportError:
            logger.debug("FEEN not available - will use software fallback")
        except Exception as e:
            logger.warning(f"FEEN initialization failed: {e} - using software fallback")
    
    def is_available(self) -> bool:
        """
        Check if FEEN hardware is available.
        
        Returns:
            True if FEEN can be used, False otherwise.
        """
        return self._feen_available and self._scorer is not None
    
    def compute(
        self,
        raw_value: float,
        peers: Optional[List[float]] = None,
        history: Optional[List[float]] = None
    ) -> ConfidenceResult:
        """
        Compute confidence using FEEN hardware.
        
        Args:
            raw_value: Candidate value to evaluate
            peers: Optional list of peer values for agreement check
            history: Optional recent history for stability/likelihood
        
        Returns:
            ConfidenceResult with score and components
        
        Raises:
            RuntimeError: If FEEN is not available (caller should check is_available())
        """
        if not self.is_available():
            raise RuntimeError(
                "FEEN not available. Check is_available() before calling compute()."
            )
        
        # Prepare inputs (handle None defaults)
        peers_list = peers if peers is not None else []
        history_list = history if history is not None else []
        
        # Call FEEN hardware
        # The C++ layer handles the physics - we just bridge the call
        feen_result = self._scorer.compute(
            raw_value=float(raw_value),
            peers=peers_list,
            history=history_list
        )
        
        # Translate FEEN result to AILEE format
        # FEEN provides: {score, stability, agreement, likelihood}
        # AILEE expects: ConfidenceResult(score, stability, agreement, likelihood)
        return ConfidenceResult(
            score=float(feen_result.score),
            stability=float(feen_result.stability),
            agreement=float(feen_result.agreement),
            likelihood=float(feen_result.likelihood)
        )
    
    def get_hardware_diagnostics(self) -> Optional[dict]:
        """
        Get FEEN hardware status for monitoring.
        
        Returns diagnostic info if FEEN is available, None otherwise.
        Used by AILEE monitoring layer to detect hardware issues.
        
        Returns:
            dict with channel_energies, etc. or None if unavailable
        """
        if not self.is_available():
            return None
        
        try:
            return {
                'channel_energies': self._scorer.get_channel_energies(),
                'hardware_available': True,
                'backend': 'feen_phononic'
            }
        except Exception as e:
            logger.warning(f"Failed to get FEEN diagnostics: {e}")
            return None
    
    def _default_config(self) -> dict:
        """
        Default FEEN hardware configuration.
        
        These values match AILEE's confidence weights and history windows.
        Changes here should be coordinated with AILEE policy.
        """
        return {
            'frequency_base_hz': 1000.0,
            'frequency_spacing_hz': 10.0,
            'q_factor': 500.0,
            
            # Weights (must match AILEE's AileeConfig defaults)
            'w_stability': 0.45,
            'w_agreement': 0.30,
            'w_likelihood': 0.25,
            
            # History window (must match AILEE's history_window)
            'history_window': 60,
            
            # Thresholds (must match AILEE's grace config)
            'agreement_delta': 0.10,
            'max_abs_z': 3.0,
        }
    
    def update_history(self, value: float) -> None:
        """
        Update FEEN's internal history buffer.
        
        AILEE is responsible for calling this after each trusted decision.
        
        Args:
            value: Trusted value to add to history
        """
        if self.is_available():
            self._scorer.update_history(float(value))


def create_feen_scorer(config: Optional[dict] = None) -> FEENConfidenceScorer:
    """
    Factory function for FEEN confidence scorer.
    
    Usage:
        scorer = create_feen_scorer()
        if scorer.is_available():
            result = scorer.compute(value, peers, history)
        else:
            # Use software fallback
    
    Args:
        config: Optional FEEN configuration dict
    
    Returns:
        FEENConfidenceScorer instance (may or may not be available)
    """
    return FEENConfidenceScorer(config)
