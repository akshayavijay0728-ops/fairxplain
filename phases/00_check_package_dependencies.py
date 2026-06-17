"""
Name: check_package_dependencies
Description: Checks availability of 11 ML/AI packages (xgboost, shap, lime, aif360, tensorflow, fairlearn, dice_ml, dowhy, imblearn, fpdf2, streamlit) and reports which are installed vs. missing.
"""

import sys

packages_to_check = [
    'xgboost', 'shap', 'lime', 'aif360', 'tensorflow',
    'fairlearn', 'dice_ml', 'dowhy', 'imblearn', 'fpdf2', 'streamlit'
]

installed = []
missing = []

for pkg in packages_to_check:
    try:
        __import__(pkg)
        installed.append(pkg)
    except ImportError:
        missing.append(pkg)

print("✅ INSTALLED:")
for pkg in installed:
    print(f"  • {pkg}")

if missing:
    print("\n❌ MISSING:")
    for pkg in missing:
        print(f"  • {pkg}")
else:
    print("\n✅ ALL PACKAGES READY!")