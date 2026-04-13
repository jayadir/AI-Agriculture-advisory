import csv
import statistics
from collections import Counter
from pathlib import Path


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "eval_results.csv"
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    print(f"rows={len(rows)}")

    numeric_fields = ["Latency_s", "Correctness", "Groundedness", "Style"]
    for field in numeric_fields:
        values = [float(r[field]) for r in rows]
        print(
            f"{field}: mean={statistics.mean(values):.2f} "
            f"median={statistics.median(values):.2f} min={min(values):.2f} max={max(values):.2f}"
        )

    for field in ["Correctness", "Groundedness", "Style"]:
        dist = Counter(int(float(r[field])) for r in rows)
        dist_str = ", ".join(f"{k}:{dist[k]}" for k in sorted(dist))
        print(f"{field}_dist: {dist_str}")

    worst = sorted(rows, key=lambda r: (float(r["Groundedness"]), float(r["Correctness"]), float(r["Latency_s"])))
    print("worst_by_groundedness:")
    for r in worst[:10]:
        print(f"  {r['ID']}: G={r['Groundedness']} C={r['Correctness']} L={r['Latency_s']}")

    buckets = [0, 30, 60, 90, 120, 180, 999]
    bucket_counts: Counter[str] = Counter()
    for r in rows:
        x = float(r["Latency_s"])
        for i in range(len(buckets) - 1):
            if buckets[i] <= x < buckets[i + 1]:
                label = f"{buckets[i]}-{buckets[i + 1]}s"
                bucket_counts[label] += 1
                break
    print("latency_buckets:")
    for k in [f"{buckets[i]}-{buckets[i + 1]}s" for i in range(len(buckets) - 1)]:
        if k in bucket_counts:
            print(f"  {k}: {bucket_counts[k]}")


if __name__ == "__main__":
    main()
