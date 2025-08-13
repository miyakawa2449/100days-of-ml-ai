# scripts/mark_today_done.py
import csv, argparse, datetime, pathlib

CSV = pathlib.Path("progress/progress.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--notes", type=str, default="")
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD（省略時はJSTの今日）")
    args = ap.parse_args()

    today = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y-%m-%d")
    date = args.date or today

    rows = []
    with CSV.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)

    hit = False
    for row in rows:
        if str(row["Day"]) == str(args.day):
            row["Date"] = date
            row["Done"] = "TRUE"
            if args.notes:
                row["Notes"] = args.notes
            hit = True
            break

    if not hit:
        raise SystemExit(f"Day {args.day} が見つかりませんでした。")

    with CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Day","Phase","Theme","Date","Done","Notes"])
        w.writeheader()
        w.writerows(rows)

    print(f"Day {args.day} を更新しました：Date={date}, Done=TRUE")

if __name__ == "__main__":
    main()
