from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

EVENT_DIRS = {
    "event_4624": INTERIM_DIR / "event_4624",
    "event_4625": INTERIM_DIR / "event_4625",
}


def normalize_event_files(event_name: str, directory: Path, dry_run: bool = True):
    """
    Rename:
        wls_day-01.parquet
    ->  event_4624_day-01.parquet

    hoặc:
        wls_day-01.parquet
    ->  event_4625_day-01.parquet

    tùy vào thư mục hiện tại.
    """

    if not directory.exists():
        print(f"[SKIP] Directory does not exist: {directory}")
        return

    pattern = re.compile(r"^wls_day-(\d+)\.parquet$")

    for file_path in sorted(directory.glob("*.parquet")):
        match = pattern.match(file_path.name)

        # File đã đúng format thì bỏ qua
        if not match:
            continue

        day = int(match.group(1))

        # Luôn dùng 2 chữ số: 1 -> 01
        new_name = f"{event_name}_day-{day:02d}.parquet"
        new_path = file_path.with_name(new_name)

        # Tránh ghi đè dữ liệu
        if new_path.exists():
            print(
                f"[CONFLICT] {file_path.name} -> {new_name} "
                "(target already exists)"
            )
            continue

        if dry_run:
            print(f"[DRY RUN] {file_path.name} -> {new_name}")
        else:
            file_path.rename(new_path)
            print(f"[RENAMED] {file_path.name} -> {new_name}")


def main():
    # Để True khi muốn kiểm tra trước.
    # Đổi thành False khi chắc chắn muốn rename.
    dry_run = False
    

    for event_name, directory in EVENT_DIRS.items():
        print(f"\n=== {event_name} ===")
        normalize_event_files(
            event_name=event_name,
            directory=directory,
            dry_run=dry_run,
        )


if __name__ == "__main__":
    main()