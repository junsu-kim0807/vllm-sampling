import torch
obj = torch.load(
    "./results/spec_decode/cost_breakdown.pt",
    map_location="cpu",
    weights_only=False,
)

mismatches = obj["validation"]["path_payload_mismatches"]
print("num mismatches:", len(mismatches))

for x in mismatches[:20]:
    print("=" * 80)
    print("record_key:", x["record_key"])
    print("source_file:", x["source_file"])
    print("mismatched_fields:", x["mismatched_fields"])