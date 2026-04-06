"""
Test Suite for Safe Pipeline Improvements
=========================================

Tests all 7 phases of the incremental improvements.

Author: Safe Refactor System
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safe_pipeline_improvements import (
    detect_column_types_enhanced,
    validate_column_consistency,
    controlled_realignment,
    detect_sections_improved,
    validate_totals_improved,
    compute_meaningful_confidence,
    detect_table_type_enhanced,
    assess_reliability,
    run_safe_pipeline,
    enhance_extraction_result,
    ColumnRole,
)


# =============================================================================
# TEST DATA
# =============================================================================

SAMPLE_FINANCIAL_TABLE = {
    "columns": ["Label", "Note", "31/12/2024", "31/12/2023", "Variation"],
    "rows": [
        {"Label": "ACTIF", "Note": "", "31/12/2024": "", "31/12/2023": "", "Variation": "", "type": "section"},
        {"Label": "Caisses et Banques", "Note": "(1)", "31/12/2024": "1 500 000", "31/12/2023": "1 200 000", "Variation": "300 000", "type": "data"},
        {"Label": "Créances clients", "Note": "(2)", "31/12/2024": "2 500 000", "31/12/2023": "2 000 000", "Variation": "500 000", "type": "data"},
        {"Label": "TOTAL ACTIF", "Note": "", "31/12/2024": "4 000 000", "31/12/2023": "3 200 000", "Variation": "800 000", "type": "total"},
    ]
}

MISALIGNED_TABLE = {
    "columns": ["Label", "Note", "31/12/2024", "31/12/2023"],
    "rows": [
        {"Label": "ACTIF", "Note": "", "31/12/2024": "", "31/12/2023": "", "type": "section"},
        # Note "(1)" is misplaced in 2024 column
        {"Label": "Caisses", "Note": "", "31/12/2024": "(1)", "31/12/2023": "1 200 000", "type": "data"},
        {"Label": "Créances", "Note": "(2)", "31/12/2024": "2 500 000", "31/12/2023": "2 000 000", "type": "data"},
    ]
}

CASH_FLOW_TABLE = {
    "columns": ["Label", "Note", "2024", "2023"],
    "rows": [
        {"Label": "FLUX DE TRESORERIE", "Note": "", "2024": "", "2023": "", "type": "section"},
        {"Label": "Encaissements", "Note": "(1)", "2024": "5 000 000", "2023": "4 500 000", "type": "data"},
        {"Label": "Décaissements", "Note": "(2)", "2024": "3 000 000", "2023": "2 800 000", "type": "data"},
    ]
}


# =============================================================================
# PHASE 1 TESTS: Column Type Detection
# =============================================================================

class TestPhase1ColumnTypeDetection:
    """Test enhanced column type detection."""
    
    def test_detects_label_column(self):
        """Should detect Label as label column."""
        result = detect_column_types_enhanced(
            SAMPLE_FINANCIAL_TABLE["rows"],
            SAMPLE_FINANCIAL_TABLE["columns"]
        )
        assert "Label" in result.columns
        assert result.columns["Label"].detected_role == ColumnRole.LABEL
    
    def test_detects_note_column(self):
        """Should detect Note as note column."""
        result = detect_column_types_enhanced(
            SAMPLE_FINANCIAL_TABLE["rows"],
            SAMPLE_FINANCIAL_TABLE["columns"]
        )
        assert "Note" in result.columns
        assert result.columns["Note"].detected_role == ColumnRole.NOTE
    
    def test_detects_date_columns(self):
        """Should detect date columns and order them."""
        result = detect_column_types_enhanced(
            SAMPLE_FINANCIAL_TABLE["rows"],
            SAMPLE_FINANCIAL_TABLE["columns"]
        )
        assert len(result.detected_date_order) >= 2
        # Most recent date should be first
        assert "31/12/2024" in result.detected_date_order[0]
    
    def test_assigns_date_roles(self):
        """Should assign current/previous roles to dates."""
        result = detect_column_types_enhanced(
            SAMPLE_FINANCIAL_TABLE["rows"],
            SAMPLE_FINANCIAL_TABLE["columns"]
        )
        current_date_col = result.detected_date_order[0]
        previous_date_col = result.detected_date_order[1]
        
        assert result.columns[current_date_col].detected_role == ColumnRole.DATE_CURRENT
        assert result.columns[previous_date_col].detected_role == ColumnRole.DATE_PREVIOUS
    
    def test_detects_variation_column(self):
        """Should detect Variation as variation amount."""
        result = detect_column_types_enhanced(
            SAMPLE_FINANCIAL_TABLE["rows"],
            SAMPLE_FINANCIAL_TABLE["columns"]
        )
        assert "Variation" in result.columns
        assert result.columns["Variation"].detected_role == ColumnRole.VARIATION_AMOUNT
    
    def test_schema_quality_score(self):
        """Should compute meaningful schema quality."""
        result = detect_column_types_enhanced(
            SAMPLE_FINANCIAL_TABLE["rows"],
            SAMPLE_FINANCIAL_TABLE["columns"]
        )
        # Good schema should have high quality
        assert result.schema_quality >= 0.6


# =============================================================================
# PHASE 2 TESTS: Safe Column Validation
# =============================================================================

class TestPhase2SafeValidation:
    """Test safe column validation without data modification."""
    
    def test_detects_note_in_numeric_column(self):
        """Should detect note patterns in numeric columns."""
        column_types = detect_column_types_enhanced(
            MISALIGNED_TABLE["rows"],
            MISALIGNED_TABLE["columns"]
        )
        
        result = validate_column_consistency(
            MISALIGNED_TABLE["rows"],
            column_types
        )
        
        # Should find the misplaced note
        assert len(result.warnings) > 0
        assert any(w.actual_type == "note_pattern" for w in result.warnings)
    
    def test_consistent_table_no_warnings(self):
        """Consistent table should have no high-severity warnings."""
        column_types = detect_column_types_enhanced(
            SAMPLE_FINANCIAL_TABLE["rows"],
            SAMPLE_FINANCIAL_TABLE["columns"]
        )
        
        result = validate_column_consistency(
            SAMPLE_FINANCIAL_TABLE["rows"],
            column_types
        )
        
        high_severity = [w for w in result.warnings if w.severity == "high"]
        assert len(high_severity) == 0
    
    def test_original_data_unchanged(self):
        """Validation should not modify original data."""
        import copy
        original = copy.deepcopy(MISALIGNED_TABLE["rows"])
        
        column_types = detect_column_types_enhanced(
            MISALIGNED_TABLE["rows"],
            MISALIGNED_TABLE["columns"]
        )
        
        validate_column_consistency(
            MISALIGNED_TABLE["rows"],
            column_types
        )
        
        # Original should be unchanged
        assert MISALIGNED_TABLE["rows"] == original


# =============================================================================
# PHASE 3 TESTS: Controlled Realignment
# =============================================================================

class TestPhase3ControlledRealignment:
    """Test controlled realignment of obvious cases only."""
    
    def test_corrects_misplaced_note(self):
        """Should move note pattern to Note column."""
        column_types = detect_column_types_enhanced(
            MISALIGNED_TABLE["rows"],
            MISALIGNED_TABLE["columns"]
        )
        validation_result = validate_column_consistency(
            MISALIGNED_TABLE["rows"],
            column_types
        )
        
        result = controlled_realignment(
            MISALIGNED_TABLE["rows"],
            column_types,
            validation_result,
            strict_mode=True
        )
        
        # Should have made corrections
        assert len(result.corrections) > 0
    
    def test_original_data_unchanged_after_realignment(self):
        """Realignment should not modify original data."""
        import copy
        original = copy.deepcopy(MISALIGNED_TABLE["rows"])
        
        column_types = detect_column_types_enhanced(
            MISALIGNED_TABLE["rows"],
            MISALIGNED_TABLE["columns"]
        )
        validation_result = validate_column_consistency(
            MISALIGNED_TABLE["rows"],
            column_types
        )
        
        controlled_realignment(
            MISALIGNED_TABLE["rows"],
            column_types,
            validation_result,
            strict_mode=True
        )
        
        # Original should be unchanged
        assert MISALIGNED_TABLE["rows"] == original


# =============================================================================
# PHASE 4 TESTS: Improved Total Validation
# =============================================================================

class TestPhase4TotalValidation:
    """Test improved section-aware total validation."""
    
    def test_detects_sections(self):
        """Should detect sections with totals."""
        sections = detect_sections_improved(SAMPLE_FINANCIAL_TABLE["rows"])
        
        assert len(sections) > 0
        # Should find the total row
        assert any(s.total_row is not None for s in sections)
    
    def test_validates_correct_total(self):
        """Should validate correct totals."""
        column_types = detect_column_types_enhanced(
            SAMPLE_FINANCIAL_TABLE["rows"],
            SAMPLE_FINANCIAL_TABLE["columns"]
        )
        
        result = validate_totals_improved(
            SAMPLE_FINANCIAL_TABLE["rows"],
            column_types
        )
        
        # 1,500,000 + 2,500,000 = 4,000,000 ✓
        assert result.validation_passed
        assert len(result.total_errors) == 0


# =============================================================================
# PHASE 5 TESTS: Meaningful Confidence
# =============================================================================

class TestPhase5MeaningfulConfidence:
    """Test meaningful confidence scoring."""
    
    def test_high_confidence_for_good_table(self):
        """Good table should have high confidence."""
        column_types = detect_column_types_enhanced(
            SAMPLE_FINANCIAL_TABLE["rows"],
            SAMPLE_FINANCIAL_TABLE["columns"]
        )
        validation_result = validate_column_consistency(
            SAMPLE_FINANCIAL_TABLE["rows"],
            column_types
        )
        total_validation = validate_totals_improved(
            SAMPLE_FINANCIAL_TABLE["rows"],
            column_types
        )
        
        confidence = compute_meaningful_confidence(
            SAMPLE_FINANCIAL_TABLE["rows"],
            column_types,
            validation_result,
            total_validation
        )
        
        assert confidence.final_score >= 0.5
    
    def test_confidence_breakdown_components(self):
        """Should provide breakdown of confidence components."""
        column_types = detect_column_types_enhanced(
            SAMPLE_FINANCIAL_TABLE["rows"],
            SAMPLE_FINANCIAL_TABLE["columns"]
        )
        validation_result = validate_column_consistency(
            SAMPLE_FINANCIAL_TABLE["rows"],
            column_types
        )
        total_validation = validate_totals_improved(
            SAMPLE_FINANCIAL_TABLE["rows"],
            column_types
        )
        
        confidence = compute_meaningful_confidence(
            SAMPLE_FINANCIAL_TABLE["rows"],
            column_types,
            validation_result,
            total_validation
        )
        
        # All components should be present
        assert hasattr(confidence, 'schema_validity')
        assert hasattr(confidence, 'column_consistency')
        assert hasattr(confidence, 'validation_pass_rate')
        assert hasattr(confidence, 'coverage')


# =============================================================================
# PHASE 6 TESTS: Table Type Detection
# =============================================================================

class TestPhase6TableTypeDetection:
    """Test enhanced table type detection."""
    
    def test_detects_cash_flow(self):
        """Should detect cash flow table."""
        result = detect_table_type_enhanced(CASH_FLOW_TABLE["rows"])
        
        assert result["table_type"] == "cash_flow"
        assert result["confidence"] >= 0.5
    
    def test_detects_balance_sheet(self):
        """Should detect balance sheet keywords."""
        result = detect_table_type_enhanced(
            SAMPLE_FINANCIAL_TABLE["rows"],
            title="BILAN"
        )
        
        # Either balance_sheet or includes ACTIF keyword
        assert result["scores"]["balance_sheet"] >= 1


# =============================================================================
# PHASE 7 TESTS: Safety Layer
# =============================================================================

class TestPhase7SafetyLayer:
    """Test reliability assessment and safety flags."""
    
    def test_reliable_table_flagged_reliable(self):
        """Good table should be flagged as reliable."""
        column_types = detect_column_types_enhanced(
            SAMPLE_FINANCIAL_TABLE["rows"],
            SAMPLE_FINANCIAL_TABLE["columns"]
        )
        validation_result = validate_column_consistency(
            SAMPLE_FINANCIAL_TABLE["rows"],
            column_types
        )
        total_validation = validate_totals_improved(
            SAMPLE_FINANCIAL_TABLE["rows"],
            column_types
        )
        confidence = compute_meaningful_confidence(
            SAMPLE_FINANCIAL_TABLE["rows"],
            column_types,
            validation_result,
            total_validation
        )
        
        reliability = assess_reliability(
            column_types,
            validation_result,
            total_validation,
            confidence
        )
        
        assert reliability.is_reliable
        assert reliability.recommended_action in ("accept", "review")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Test complete pipeline integration."""
    
    def test_run_safe_pipeline_complete(self):
        """Should run all phases successfully."""
        result = run_safe_pipeline(SAMPLE_FINANCIAL_TABLE)
        
        # All phases should complete
        assert result.column_types is not None
        assert result.validation_result is not None
        assert result.total_validation is not None
        assert result.confidence is not None
        assert result.reliability is not None
        assert result.metadata is not None
    
    def test_enhance_extraction_result(self):
        """Should enhance existing result with metadata."""
        import copy
        data = copy.deepcopy(SAMPLE_FINANCIAL_TABLE)
        
        enhanced = enhance_extraction_result(data)
        
        # Should have new metadata
        assert "_safe_pipeline" in enhanced
        assert "_confidence" in enhanced
        assert "_is_reliable" in enhanced
    
    def test_graceful_failure_on_bad_input(self):
        """Should handle bad input gracefully."""
        bad_data = {"not": "valid"}
        
        result = enhance_extraction_result(bad_data)
        
        # Should return original data (with maybe an error note)
        assert result == bad_data or "_safe_pipeline_error" in result


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SAFE PIPELINE IMPROVEMENTS - TEST SUITE")
    print("=" * 70)
    
    # Run with pytest
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    
    print("\n" + "=" * 70)
    if exit_code == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)
