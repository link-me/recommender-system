import argparse
from pathlib import Path
from recommender import load_interactions, build_user_item_matrix, recommend_for_user, popular_items


def parse_args():
    p = argparse.ArgumentParser(description="Simple recommender system demo (Pandas)")
    p.add_argument("--data", required=True, help="Path to CSV with columns: user_id,item_id,rating")
    p.add_argument("--user", required=False, help="Target user_id to recommend for")
    p.add_argument("--top", type=int, default=5, help="Top-N recommendations")
    p.add_argument("--fallback", action="store_true", help="Return popular items if user not provided")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.data)
    data = load_interactions(str(csv_path))

    if args.user:
        mat, users, items = build_user_item_matrix(data)
        recs = recommend_for_user(args.user, mat, users, items, top_n=args.top)
        print({"user": args.user, "recommendations": recs})
    else:
        if not args.fallback:
            raise SystemExit("Provide --user or use --fallback to show popular items")
        recs = popular_items(data, top_n=args.top)
        print({"popular": recs})


if __name__ == "__main__":
    main()
