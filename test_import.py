#!/usr/bin/env python
"""Test import of column_realignment_engine module"""

try:
    from column_realignment_engine import (
        detect_column_types,
        realign_columns,
        run_column_realignment,
        detect_cashflow_table
    )
    print("✅ Column realignment engine imported successfully")

    # Also test the import chain
    from json_table_utils import post_process_extraction
    print("✅ Full import chain works")
    
    print("\n✅✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()
