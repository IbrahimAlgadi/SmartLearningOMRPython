import json
d = json.load(open("detected-test_blank_ar/result.json"))
det = d.get("answer_details", {})
print(f"{'Q':>4}  {'choice':>6}  {'fill':>6}  {'mean':>6}  {'tier':>8}  note")
print("-" * 55)
for q in range(1, 6):
    k = str(q)
    v = det.get(k, {})
    print(f"Q{q:>3}  {str(v.get('choice','-')):>6}  "
          f"{str(v.get('fill', v.get('row_max_fill', '-'))):>6}  "
          f"{str(v.get('row_mean_fill','-')):>6}  "
          f"{str(v.get('tier','')):>8}  {v.get('note','')}")
