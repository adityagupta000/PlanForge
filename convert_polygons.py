import json
from pathlib import Path

def flatten_polygon_file(file_path: Path):
    with open(file_path, "r") as f:
        data = json.load(f)

    flat_list = []

    # If file already flat, skip
    if isinstance(data, list):
        print(f"[SKIP] Already flat: {file_path}")
        return

    # Otherwise, flatten categories
    for category in ["walls", "doors", "windows", "floors", "fixtures"]:
        if category in data and isinstance(data[category], list):
            for idx, poly in enumerate(data[category]):
                flat_list.append({
                    "id": idx,
                    "type": category[:-1],  # "walls" -> "wall"
                    "points": poly.get("points", []),
                    "area": poly.get("area", None)
                })

    # Save back in flat format
    with open(file_path, "w") as f:
        json.dump(flat_list, f, indent=2)

    print(f"[OK] Converted: {file_path}")


def batch_convert(root_dir="data/floorplans"):
    root = Path(root_dir)
    for polygon_file in root.rglob("polygon.json"):
        flatten_polygon_file(polygon_file)


if __name__ == "__main__":
    # Change path if needed
    batch_convert("data/floorplans")
