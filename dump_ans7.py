import json

d = json.load(open("detected-00_original/result.json"))
det = d.get("answer_details", {})
print(f"{'Q':>4}  {'choice':>6}  {'fill':>6}  {'mean':>6}  {'tier':>8}  note")
print("-" * 55)

double_marks = []
unanswered = []

for q in range(1, 101):
    k = str(q)
    v = det.get(k, {})
    note = v.get('note', '')
    if note == 'double_mark':
        double_marks.append((q, v))
    elif note == 'unanswered':
        unanswered.append((q, v))

print("\n--- Double Marked ---")
for q, v in double_marks:
    print(f"Q{q:>3}  {str(v.get('choice','-')):>6}  "
          f"{str(v.get('fill', v.get('row_max_fill', '-'))):>6}  "
          f"{str(v.get('row_mean_fill','-')):>6}  "
          f"{str(v.get('tier','')):>8}  {v.get('note','')}")

print("\n--- Unanswered ---")
for q, v in unanswered:
    print(f"Q{q:>3}  {str(v.get('choice','-')):>6}  "
          f"{str(v.get('fill', v.get('row_max_fill', '-'))):>6}  "
          f"{str(v.get('row_mean_fill','-')):>6}  "
          f"{str(v.get('tier','')):>8}  {v.get('note','')}")
