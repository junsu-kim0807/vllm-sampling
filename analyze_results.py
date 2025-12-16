#!/usr/bin/env python3
"""
벤치마크 결과 분석 및 비교 스크립트
"""

import json
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

def load_json_results(filepath: str) -> Dict:
    """JSON 결과 파일 로드"""
    with open(filepath) as f:
        return json.load(f)

def compare_results(vanilla_file: str, starkv_file: str) -> Dict:
    """Vanilla와 StarKV 결과 비교"""
    vanilla = load_json_results(vanilla_file)
    starkv = load_json_results(starkv_file)
    
    comparison = {
        "scenario": Path(vanilla_file).stem,
        "vanilla": {
            "rps": vanilla.get("requests_per_second", 0),
            "tps": vanilla.get("tokens_per_second", 0),
            "elapsed_time": vanilla.get("elapsed_time", 0),
        },
        "starkv": {
            "rps": starkv.get("requests_per_second", 0),
            "tps": starkv.get("tokens_per_second", 0),
            "elapsed_time": starkv.get("elapsed_time", 0),
            "reforward_requests": starkv.get("starkv_stats", {}).get("total_reforward_requests", 0),
            "reforward_tokens": starkv.get("starkv_stats", {}).get("total_reforward_tokens", 0),
        },
    }
    
    # 개선률 계산
    if comparison["vanilla"]["rps"] > 0:
        rps_improvement = ((comparison["starkv"]["rps"] / comparison["vanilla"]["rps"]) - 1) * 100
    else:
        rps_improvement = 0
    
    if comparison["vanilla"]["tps"] > 0:
        tps_improvement = ((comparison["starkv"]["tps"] / comparison["vanilla"]["tps"]) - 1) * 100
    else:
        tps_improvement = 0
    
    comparison["improvement"] = {
        "rps_percent": rps_improvement,
        "tps_percent": tps_improvement,
    }
    
    # Reforward rate 계산
    total_requests = starkv.get("num_requests", 0)
    total_tokens = starkv.get("total_num_tokens", 0)
    if total_requests > 0:
        comparison["starkv"]["reforward_request_rate"] = (
            comparison["starkv"]["reforward_requests"] / total_requests * 100
        )
    if total_tokens > 0:
        comparison["starkv"]["reforward_token_rate"] = (
            comparison["starkv"]["reforward_tokens"] / total_tokens * 100
        )
    
    return comparison

def print_comparison(comparison: Dict):
    """비교 결과 출력"""
    print(f"\n{'='*60}")
    print(f"Scenario: {comparison['scenario']}")
    print(f"{'='*60}")
    
    print("\nVanilla vLLM:")
    print(f"  RPS: {comparison['vanilla']['rps']:.2f}")
    print(f"  TPS: {comparison['vanilla']['tps']:.2f}")
    print(f"  Elapsed: {comparison['vanilla']['elapsed_time']:.2f}s")
    
    print("\nStarKV:")
    print(f"  RPS: {comparison['starkv']['rps']:.2f}")
    print(f"  TPS: {comparison['starkv']['tps']:.2f}")
    print(f"  Elapsed: {comparison['starkv']['elapsed_time']:.2f}s")
    print(f"  Reforward Requests: {comparison['starkv']['reforward_requests']}")
    print(f"  Reforward Tokens: {comparison['starkv']['reforward_tokens']}")
    if 'reforward_request_rate' in comparison['starkv']:
        print(f"  Reforward Request Rate: {comparison['starkv']['reforward_request_rate']:.2f}%")
    if 'reforward_token_rate' in comparison['starkv']:
        print(f"  Reforward Token Rate: {comparison['starkv']['reforward_token_rate']:.2f}%")
    
    print("\nImprovement:")
    rps_imp = comparison['improvement']['rps_percent']
    tps_imp = comparison['improvement']['tps_percent']
    print(f"  RPS: {rps_imp:+.2f}%")
    print(f"  TPS: {tps_imp:+.2f}%")
    
    if rps_imp > 0:
        print(f"  ✓ StarKV shows {rps_imp:.2f}% RPS improvement")
    else:
        print(f"  ✗ StarKV shows {abs(rps_imp):.2f}% RPS degradation")

def analyze_csv(csv_file: str):
    """CSV 결과 파일 분석"""
    df = pd.read_csv(csv_file)
    
    print("\n" + "="*60)
    print("Summary Analysis")
    print("="*60)
    
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        vanilla = scenario_df[scenario_df['variant'] == 'vllm']
        starkv = scenario_df[scenario_df['variant'] == 'starkv']
        
        if len(vanilla) > 0 and len(starkv) > 0:
            v_rps = float(vanilla['requests_per_second'].values[0])
            s_rps = float(starkv['requests_per_second'].values[0])
            v_tps = float(vanilla['tokens_per_second'].values[0])
            s_tps = float(starkv['tokens_per_second'].values[0])
            
            if v_rps > 0:
                rps_imp = ((s_rps / v_rps) - 1) * 100
            else:
                rps_imp = 0
            
            if v_tps > 0:
                tps_imp = ((s_tps / v_tps) - 1) * 100
            else:
                tps_imp = 0
            
            print(f"\n{scenario}:")
            print(f"  RPS: {v_rps:.2f} -> {s_rps:.2f} ({rps_imp:+.2f}%)")
            print(f"  TPS: {v_tps:.2f} -> {s_tps:.2f} ({tps_imp:+.2f}%)")
            print(f"  Reforward: {starkv['starkv_reforward_requests'].values[0]} requests, "
                  f"{starkv['starkv_reforward_tokens'].values[0]} tokens")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analyze_results.py <vanilla.json> <starkv.json>  # Compare two JSON files")
        print("  python analyze_results.py <results.csv>                 # Analyze CSV file")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # CSV 분석
        analyze_csv(sys.argv[1])
    elif len(sys.argv) == 3:
        # 두 JSON 파일 비교
        comparison = compare_results(sys.argv[1], sys.argv[2])
        print_comparison(comparison)
    else:
        print("Invalid arguments")
        sys.exit(1)

if __name__ == "__main__":
    main()

