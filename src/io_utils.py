import json
from pathlib import Path


def parse_list(text: str, prefix: str = "-") -> list[str]:
    return [
        item.strip().lstrip(prefix).strip()
        for item in text.splitlines()
        if item.strip()
    ]


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_json(
    path: str | Path,
    data,
    make_dir: bool = True,
    **kwargs,
) -> None:
    path = Path(path)

    if make_dir:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            **kwargs,
        )


def save_jsonl(
    path: str | Path,
    data: list[dict],
    make_dir: bool = True,
) -> None:
    path = Path(path)

    if make_dir:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for row in data:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )
