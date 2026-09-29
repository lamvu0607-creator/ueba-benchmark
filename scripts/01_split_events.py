"""
01_split_events.py

Tách dữ liệu LANL theo từng ngày:

    EventID = 4624 -> ../data/interim/events_4624/
    EventID = 4625 -> ../data/interim/events_4625/

Mỗi file đầu vào sinh tối đa 2 file đầu ra tương ứng.

Bước này chỉ filter EventID.
Không thực hiện cleaning hay feature engineering.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# %% CONFIG

INPUT_DIR = Path("../data/parquet")

OUTPUT_4624_DIR = Path(
    "../data/interim/event_4624"
)

OUTPUT_4625_DIR = Path(
    "../data/interim/event_4625"
)

START_DAY = 1
END_DAY = 15

MIN_DAY = 1
MAX_DAY = 30

INPUT_FILE_TEMPLATE = "wls_day-{day:02d}.parquet"
OUTPUT_FILE_TEMPLATE = "wls_day-{day:02d}.parquet"

BATCH_SIZE = 250_000
OVERWRITE = False

EVENT_SUCCESS = 4624
EVENT_FAILURE = 4625


# %% VALIDATION

def validate_config() -> None:
    if not (
        MIN_DAY
        <= START_DAY
        <= END_DAY
        <= MAX_DAY
    ):
        raise ValueError(
            "Khoảng ngày không hợp lệ: "
            f"{START_DAY} -> {END_DAY}. "
            f"Phải nằm trong {MIN_DAY} -> {MAX_DAY}."
        )


def get_input_path(day: int) -> Path:
    return INPUT_DIR / INPUT_FILE_TEMPLATE.format(
        day=day
    )


def get_output_paths(
    day: int,
) -> tuple[Path, Path]:
    filename = OUTPUT_FILE_TEMPLATE.format(
        day=day
    )

    return (
        OUTPUT_4624_DIR / filename,
        OUTPUT_4625_DIR / filename,
    )


def prepare_output(path: Path) -> None:
    if not path.exists():
        return

    if not OVERWRITE:
        raise FileExistsError(
            f"Output đã tồn tại: {path}"
        )

    path.unlink()


def get_event_values(
    event_type: pa.DataType,
) -> tuple[int | str, int | str]:

    if pa.types.is_integer(event_type):
        return EVENT_SUCCESS, EVENT_FAILURE

    if (
        pa.types.is_string(event_type)
        or pa.types.is_large_string(event_type)
    ):
        return (
            str(EVENT_SUCCESS),
            str(EVENT_FAILURE),
        )

    raise TypeError(
        "EventID phải là integer hoặc string. "
        f"Kiểu hiện tại: {event_type}"
    )


def validate_schema(
    current_schema: pa.Schema,
    reference_schema: pa.Schema,
    day: int,
) -> None:

    if current_schema != reference_schema:
        raise ValueError(
            f"Schema ngày {day} khác schema chuẩn."
        )


# %% SPLIT ONE DAY

def split_one_day(
    day: int,
    reference_schema: pa.Schema | None,
) -> tuple[pa.Schema, int, int]:

    input_path = get_input_path(day)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: {input_path}"
        )

    output_4624, output_4625 = (
        get_output_paths(day)
    )

    prepare_output(output_4624)
    prepare_output(output_4625)

    dataset = ds.dataset(
        str(input_path),
        format="parquet",
    )

    if "EventID" not in dataset.schema.names:
        raise ValueError(
            f"Ngày {day} không có cột EventID."
        )

    if reference_schema is None:
        reference_schema = dataset.schema
    else:
        validate_schema(
            current_schema=dataset.schema,
            reference_schema=reference_schema,
            day=day,
        )

    event_type = dataset.schema.field(
        "EventID"
    ).type

    event_4624_value, event_4625_value = (
        get_event_values(event_type)
    )

    event_filter = ds.field(
        "EventID"
    ).isin(
        [
            event_4624_value,
            event_4625_value,
        ]
    )

    scanner = dataset.scanner(
        filter=event_filter,
        batch_size=BATCH_SIZE,
    )

    count_4624 = 0
    count_4625 = 0

    writer_4624 = pq.ParquetWriter(
        output_4624,
        reference_schema,
        compression="snappy",
    )

    writer_4625 = pq.ParquetWriter(
        output_4625,
        reference_schema,
        compression="snappy",
    )

    try:
        for batch in scanner.to_batches():

            table = pa.Table.from_batches(
                [batch]
            )

            mask_4624 = pc.equal(
                table["EventID"],
                pa.scalar(
                    event_4624_value,
                    type=event_type,
                ),
            )

            table_4624 = table.filter(
                mask_4624
            )

            if table_4624.num_rows:
                writer_4624.write_table(
                    table_4624
                )

                count_4624 += (
                    table_4624.num_rows
                )

            mask_4625 = pc.equal(
                table["EventID"],
                pa.scalar(
                    event_4625_value,
                    type=event_type,
                ),
            )

            table_4625 = table.filter(
                mask_4625
            )

            if table_4625.num_rows:
                writer_4625.write_table(
                    table_4625
                )

                count_4625 += (
                    table_4625.num_rows
                )

    finally:
        writer_4624.close()
        writer_4625.close()

    return (
        reference_schema,
        count_4624,
        count_4625,
    )


# %% MAIN

def split_events() -> None:

    validate_config()

    OUTPUT_4624_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_4625_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_days = END_DAY - START_DAY + 1

    total_4624 = 0
    total_4625 = 0

    reference_schema = None

    for index, day in enumerate(
        range(START_DAY, END_DAY + 1),
        start=1,
    ):

        (
            reference_schema,
            day_4624,
            day_4625,
        ) = split_one_day(
            day=day,
            reference_schema=reference_schema,
        )

        total_4624 += day_4624
        total_4625 += day_4625

        progress = index / total_days * 100

        print(
            f"[{progress:6.2f}%] "
            f"day {day:02d} | "
            f"4624: {day_4624:,} | "
            f"4625: {day_4625:,}"
        )

    print(
        f"\nDone | "
        f"4624: {total_4624:,} | "
        f"4625: {total_4625:,}"
    )


# %% ENTRY POINT

if __name__ == "__main__":
    split_events()