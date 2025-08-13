
import csv, datetime, re
from pathlib import Path

CSV_PATH = Path("progress/progress.csv")
README_PATH = Path("README.md")

def load_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    for row in rows:
        row["Done"] = str(row["Done"]).strip().upper() in ("TRUE","1","YES","Y","✓","CHECKED")
    return rows

def progress_bar(pct, width=40):
    filled = int(round(width * pct))
    return "█" * filled + "░" * (width - filled)

def make_table(rows):
    header = "| Day | フェーズ | 学習テーマ | 日付 | 完了 | メモ |"
    sep = "|---:|:--:|---|---:|:--:|---|"
    lines = [header, sep]
    for row in rows:
        lines.append(f"| {row['Day']} | {row.get('Phase','')} | {row['Theme']} | {row['Date']} | {'✅' if row['Done'] else '☐'} | {row['Notes']} |")
    return "\n".join(lines)

def main():
    rows = load_rows()
    total = len(rows)
    done_cnt = sum(r["Done"] for r in rows)
    pct = done_cnt / total if total else 0.0
    bar = progress_bar(pct)
    today = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y-%m-%d")  # JST

    summary = []
    summary.append(f"- 今日（JST）: **{today}**")
    summary.append(f"- 完了: **{done_cnt}/{total} ({pct*100:.1f}%)**")
    summary.append("")
    summary.append(f"```\n{bar}\n```")
    summary.append("")
    summary.append("### 進捗チェックリスト")
    summary.append(make_table(rows))
    block = "\n".join(summary)

    text = README_PATH.read_text(encoding="utf-8")
    new_text = re.sub(
        r"(<!--progress:start-->)(.*?)(<!--progress:end-->)",
        lambda m: m.group(1) + "\n" + block + "\n" + m.group(3),
        text,
        flags=re.DOTALL
    )
    if new_text != text:
        README_PATH.write_text(new_text, encoding="utf-8")
        print("README updated.")
    else:
        print("No changes.")

if __name__ == "__main__":
    main()
