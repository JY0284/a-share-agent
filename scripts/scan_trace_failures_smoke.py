import json
from pathlib import Path

TRACE_DIR = Path("traces")


def is_failed_python_tool_event(obj: dict) -> bool:
    text = json.dumps(obj, ensure_ascii=False)
    if "tool_execute_python" not in text:
        return False
    if '"success": false' in text or "'success': False" in text:
        return True
    if "Traceback" in text and "error" in text:
        return True
    return False


def main() -> None:
    failures = []
    paths = sorted(TRACE_DIR.glob("*.jsonl"))
    for p in paths[:5]:
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= 300:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if is_failed_python_tool_event(obj):
                        failures.append((p.name, i + 1))
                        break
        except Exception:
            continue

    print({
        "checked_files": len(paths[:5]),
        "files_with_fail_event_found": len(failures),
        "examples": failures[:3],
    })


if __name__ == "__main__":
    main()
