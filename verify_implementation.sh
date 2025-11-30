#!/bin/bash
# Phase 1-4 実装ファイルの検証スクリプト

echo ""
echo "============================================================"
echo "Phase 1-4 実装ファイルの検証"
echo "============================================================"
echo ""

# カウンター
total=0
exists=0
missing=0

check_file() {
    total=$((total + 1))
    if [ -f "$1" ]; then
        echo "✓ $1"
        exists=$((exists + 1))
    else
        echo "✗ $1 (存在しません)"
        missing=$((missing + 1))
    fi
}

echo "Phase 1: オフライン検証環境"
echo "----------------------------------------"
check_file "core/image_loader.py"
check_file "config/default.yaml"
check_file "ui/offline_viewer_ui.py"
check_file "main_offline_test.py"

echo ""
echo "Phase 2: 深度推定"
echo "----------------------------------------"
check_file "core/depth_estimation.py"
check_file "ui/depth_test_ui.py"
check_file "main_depth_test.py"
check_file "scripts/convert_depth_model.py"

echo ""
echo "Phase 3: ライン検出"
echo "----------------------------------------"
check_file "core/line_detection.py"
check_file "core/calibration.py"
check_file "ui/line_test_ui.py"
check_file "ui/calibration_ui.py"
check_file "main_line_test.py"
check_file "main_calibration.py"

echo ""
echo "Phase 4: 統合判定"
echo "----------------------------------------"
check_file "core/driving_decision.py"
check_file "test_phase4_integration.py"

echo ""
echo "テスト・ドキュメント"
echo "----------------------------------------"
check_file "test_components.py"
check_file "PHASE4_COMPLETION_REPORT.md"

echo ""
echo "============================================================"
echo "検証結果"
echo "============================================================"
echo "総ファイル数: $total"
echo "存在: $exists"
echo "不足: $missing"
echo ""

if [ $missing -eq 0 ]; then
    echo "✅ すべてのファイルが正常に配置されています"
    exit 0
else
    echo "⚠️ 一部のファイルが不足しています"
    exit 1
fi
