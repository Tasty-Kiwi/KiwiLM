from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / "scripts" / "colab_transformer_750k.py"
LAUNCHER = ROOT / "scripts" / "run_colab_transformer_750k.sh"


def test_transformer_750k_workload_contract() -> None:
    source = WORKLOAD.read_text(encoding="utf-8")

    ast.parse(source)
    assert "EXPECTED_PARAMETER_COUNT = 5_264_896" in source
    assert '"--architecture",' in source
    assert '"transformer",' in source
    assert '"--max-tokens",' in source
    assert '"160465920",' in source
    assert '"--warmup-tokens",' in source
    assert '"8023296",' in source
    assert "DriveBackupMonitor" in source
    assert "VerifiedDirectoryBackup" in source
    assert '"complete": True' in source
    assert "backup_monitor.backup.verify()" in source
    assert "BACKUP_MANIFEST_COPY" in source


def test_transformer_750k_launcher_recovery_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'session_name="${COLAB_SESSION_NAME:-kiwilm-transformer-750k}"' in source
    assert "scripts/validate_model_f_data.py" in source
    assert "split -b 33554432" in source
    assert '"${colab_bin}" drivemount' in source
    assert "download_colab_file()" in source
    assert "for attempt in 1 2 3; do" in source
    assert "preserve_session=1" in source
    assert "preserve_session=0" in source
    assert "Verified Google Drive backup" in source
