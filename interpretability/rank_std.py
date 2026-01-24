#!/usr/bin/env python3
"""
Analyze PMC features across multiple JSON files to find the most stable events.
Averages CV% across all files for each function/event combination.

Usage: 
    python3 rank_std.py [json_directory]
    python3 rank_std.py test/features/           # Analyze all JSON files in directory
"""

import json
import sys
import os
import glob
import math
from collections import defaultdict

def calculate_cv(mean, std):
    """Calculate coefficient of variation (CV = std/mean * 100)."""
    if mean == 0:
        return float('inf')
    return (std / mean) * 100

def extract_from_file(json_file):
    """Extract all CV values from a single JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = []
    for func_name, func_data in data.items():
        for event_id, event_data in func_data.items():
            if isinstance(event_data, dict) and 'stats' in event_data:
                stats = event_data['stats']
                if 'total_count_std' in stats and 'total_count_mean' in stats:
                    cv = calculate_cv(stats['total_count_mean'], stats['total_count_std'])
                    results.append({
                        'function': func_name,
                        'event_id': event_id,
                        'event_name': event_data.get('event_name', 'unknown'),
                        'total_count_std': stats['total_count_std'],
                        'total_count_mean': stats.get('total_count_mean', 0),
                        'cv': cv,
                        'num_samples': event_data.get('num_samples', 0)
                    })
    return results

def main():
    # Default path
    default_path = "test/features/"
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = default_path
    
    # Find all JSON files
    if os.path.isdir(input_path):
        json_files = sorted(glob.glob(os.path.join(input_path, "pmc_features_*.json")))
        output_dir = input_path
    elif os.path.isfile(input_path):
        json_files = [input_path]
        output_dir = os.path.dirname(input_path) or "."
    else:
        print(f"Error: '{input_path}' not found.")
        sys.exit(1)
    
    if not json_files:
        print(f"Error: No JSON files found in '{input_path}'")
        sys.exit(1)
    
    print(f"Analyzing {len(json_files)} JSON file(s):")
    for f in json_files:
        print(f"  - {os.path.basename(f)}")
    print()
    
    # Aggregate data across all files
    # Structure: (function, event_name) -> [list of CVs from each file]
    func_event_cvs = defaultdict(list)
    # Structure: event_name -> [list of all CVs]
    event_all_cvs = defaultdict(list)
    # Structure: event_name -> [list of all counts]
    event_all_counts = defaultdict(list)
    # Track all functions
    all_functions = set()
    
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        results = extract_from_file(json_file)
        
        for r in results:
            key = (r['function'], r['event_name'])
            func_event_cvs[key].append(r['cv'])
            event_all_cvs[r['event_name']].append({
                'cv': r['cv'],
                'function': r['function'],
                'file': file_name
            })
            event_all_counts[r['event_name']].append(r['total_count_mean'])
            all_functions.add(r['function'])
    
    num_files = len(json_files)
    num_functions = len(all_functions)
    
    print(f"Total files: {num_files}")
    print(f"Total functions: {num_functions}")
    print(f"Total events: {len(event_all_cvs)}")
    print()
    
    # ========== ANALYSIS 1: Per-function event ranking (averaged across files) ==========
    print("=" * 120)
    print("ANALYSIS 1: PER-FUNCTION EVENT RANKING (Averaged Across All Files)")
    print("=" * 120)
    
    # Group by function
    func_events = defaultdict(list)
    for (func_name, event_name), cvs in func_event_cvs.items():
        avg_cv = sum(cvs) / len(cvs)
        std_cv = (sum((x - avg_cv)**2 for x in cvs) / len(cvs)) ** 0.5 if len(cvs) > 1 else 0
        func_events[func_name].append({
            'event_name': event_name,
            'avg_cv': avg_cv,
            'std_cv': std_cv,
            'min_cv': min(cvs),
            'max_cv': max(cvs),
            'num_files': len(cvs)
        })
    
    # CSV for per-function rankings
    csv_lines = ["function,rank,event_name,avg_cv,std_cv,min_cv,max_cv,num_files"]
    
    for func_name in sorted(func_events.keys()):
        events = func_events[func_name]
        # Rank by average CV (low to high)
        ranked = sorted(events, key=lambda x: x['avg_cv'])
        
        print(f"\n--- {func_name} ---")
        print(f"{'Rank':<6} {'Avg CV%':<12} {'Std CV%':<12} {'Min CV%':<10} {'Max CV%':<10} {'#Files':<8} {'Event'}")
        print("-" * 110)
        
        for i, e in enumerate(ranked[:10], 1):  # Show top 10 per function
            print(f"{i:<6} {e['avg_cv']:<12.2f} {e['std_cv']:<12.2f} {e['min_cv']:<10.2f} {e['max_cv']:<10.2f} {e['num_files']:<8} {e['event_name']}")
            csv_lines.append(f"{func_name},{i},{e['event_name']},{e['avg_cv']:.2f},{e['std_cv']:.2f},{e['min_cv']:.2f},{e['max_cv']:.2f},{e['num_files']}")
    
    # Save per-function CSV
    per_func_csv = os.path.join(output_dir, "per_function_event_ranking.csv")
    with open(per_func_csv, 'w') as f:
        f.write('\n'.join(csv_lines))
    print(f"\nPer-function rankings saved to: {per_func_csv}")
    
    # ========== ANALYSIS 2: Global event ranking (across all functions and files) ==========
    print(f"\n{'=' * 120}")
    print("ANALYSIS 2: TOP 15 EVENTS (Aggregated Across ALL Files and Functions)")
    print("=" * 120)
    
    event_rankings = []
    for event_name, cv_data in event_all_cvs.items():
        cvs = [x['cv'] for x in cv_data]
        counts = event_all_counts[event_name]
        event_rankings.append({
            'event_name': event_name,
            'mean_cv': sum(cvs) / len(cvs),
            'median_cv': sorted(cvs)[len(cvs) // 2],
            'std_cv': (sum((x - sum(cvs)/len(cvs))**2 for x in cvs) / len(cvs)) ** 0.5,
            'max_cv': max(cvs),
            'min_cv': min(cvs),
            'count': len(cvs),
            'mean_count': sum(counts) / len(counts) if counts else 0
        })
    
    event_rankings.sort(key=lambda x: x['mean_cv'])
    
    print(f"\n{'Rank':<6} {'Mean CV%':<12} {'Median CV%':<12} {'Std CV%':<12} {'Min CV%':<10} {'Max CV%':<10} {'Count':<8} {'Event'}")
    print("-" * 120)
    
    for i, e in enumerate(event_rankings[:15], 1):
        print(f"{i:<6} {e['mean_cv']:<12.2f} {e['median_cv']:<12.2f} {e['std_cv']:<12.2f} {e['min_cv']:<10.2f} {e['max_cv']:<10.2f} {e['count']:<8} {e['event_name']}")
    
    # ========== ANALYSIS 3: Event consistency across functions ==========
    print(f"\n{'=' * 120}")
    print("ANALYSIS 3: EVENT CONSISTENCY ACROSS FUNCTIONS")
    print("(Lower std = more consistent performance across different functions)")
    print("=" * 120)
    
    # For each event, get the average CV per function, then compute std across functions
    event_func_avg = defaultdict(dict)
    for (func_name, event_name), cvs in func_event_cvs.items():
        event_func_avg[event_name][func_name] = sum(cvs) / len(cvs)
    
    event_consistency = []
    for event_name, func_avgs in event_func_avg.items():
        avg_cvs = list(func_avgs.values())
        if len(avg_cvs) > 1:
            mean_of_avgs = sum(avg_cvs) / len(avg_cvs)
            std_of_avgs = (sum((x - mean_of_avgs)**2 for x in avg_cvs) / len(avg_cvs)) ** 0.5
            event_consistency.append({
                'event_name': event_name,
                'mean_cv': mean_of_avgs,
                'std_across_funcs': std_of_avgs,
                'num_functions': len(avg_cvs)
            })
    
    event_consistency.sort(key=lambda x: x['std_across_funcs'])
    
    print(f"\n{'Rank':<6} {'Mean CV%':<12} {'Std Across Funcs':<18} {'#Funcs':<10} {'Event'}")
    print("-" * 100)
    
    for i, e in enumerate(event_consistency[:15], 1):
        print(f"{i:<6} {e['mean_cv']:<12.2f} {e['std_across_funcs']:<18.2f} {e['num_functions']:<10} {e['event_name']}")
    
    # ========== ANALYSIS 4: Event ranking by coverage ==========
    print(f"\n{'=' * 120}")
    print("ANALYSIS 4: EVENT RANKING BY COVERAGE")
    print("(Events that appear most consistently across all functions and files)")
    print("=" * 120)
    
    # Rank events by coverage
    max_coverage = num_functions * num_files
    coverage_rankings = []
    for e in event_rankings:
        coverage = e['count']
        coverage_pct = (coverage / max_coverage) * 100
        coverage_rankings.append({
            'event_name': e['event_name'],
            'coverage': coverage,
            'coverage_pct': coverage_pct,
            'mean_cv': e['mean_cv'],
            'mean_count': e['mean_count']
        })
    
    coverage_rankings.sort(key=lambda x: x['coverage'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Coverage':<12} {'Cov%':<10} {'Mean CV%':<12} {'Mean Count':<15} {'Event'}")
    print("-" * 120)
    
    for i, e in enumerate(coverage_rankings[:20], 1):  # Show top 20
        print(f"{i:<6} {e['coverage']}/{max_coverage:<9} {e['coverage_pct']:<10.1f} {e['mean_cv']:<12.2f} {e['mean_count']:<15.0f} {e['event_name']}")
    
    # Save coverage rankings
    coverage_csv = os.path.join(output_dir, "events_by_coverage.csv")
    with open(coverage_csv, 'w') as f:
        f.write("rank,event_name,coverage,coverage_pct,mean_cv,mean_count\n")
        for i, e in enumerate(coverage_rankings, 1):
            f.write(f"{i},{e['event_name']},{e['coverage']},{e['coverage_pct']:.2f},{e['mean_cv']:.2f},{e['mean_count']:.2f}\n")
    print(f"\nCoverage rankings saved to: {coverage_csv}")
    
    # ========== FINAL: Combined score ==========
    print(f"\n{'=' * 120}")
    print("FINAL: TOP 15 EVENT CANDIDATES (Combined Score)")
    print("=" * 120)
    
    # Create combined score
    event_consistency_map = {e['event_name']: e for e in event_consistency}
    
    # Calculate max coverage for normalization
    max_coverage = num_functions * num_files
    
    # Adaptive parameters based on data
    consistency_scale = num_functions  # Scale consistency penalty by number of functions
    coverage_scale = max_coverage / 10  # Scale coverage factor by max possible coverage
    
    print("Score Calculation:")
    print(f"  Score = [(mean_cv + 1) × consistency_penalty × count_factor] × coverage_penalty²")
    print(f"  Where:")
    print(f"    • mean_cv + 1: Prevents CV=0 from dominating; ensures coverage matters")
    print(f"    • consistency_penalty = (1 + std_across_funcs/{consistency_scale})")
    print(f"    • count_factor = 1/log10(mean_count)  [rewards higher counts]")
    print(f"    • coverage_penalty² = (1 + (1 - coverage_norm)²)²  [STRONG penalty for low coverage]")
    print(f"    • coverage_norm = coverage/{max_coverage}")
    print(f"")
    print(f"  Data-driven parameters:")
    print(f"    - Number of files: {num_files}")
    print(f"    - Number of functions: {num_functions}")
    print(f"    - Max coverage (functions × files): {max_coverage}")
    print(f"  Lower score = better candidate\n")
    
    combined_scores = []
    for e in event_rankings:
        consistency = event_consistency_map.get(e['event_name'], {})
        std_across = consistency.get('std_across_funcs', e['std_cv'])
        mean_count = e['mean_count']
        coverage = e['count']  # Number of (function × file) combinations
        
        # Score components (lower is better):
        # 1. Base: (mean_cv + 1) - prevents CV=0 from dominating
        # 2. Consistency penalty: scales with number of functions
        # 3. Count factor: 1/log10(count) - rewards higher counts
        # 4. Coverage penalty²: STRONG penalty for low coverage
        
        # Consistency penalty: adaptive based on number of functions
        consistency_penalty = 1.0 + (std_across / consistency_scale)
        
        # Count factor: logarithmic scale (higher counts get lower factor)
        if mean_count > 1:
            count_factor = 1.0 / math.log10(mean_count)
        else:
            count_factor = 10.0  # Heavy penalty for very low counts
        
        # Coverage penalty: SQUARED to give it more weight
        coverage_norm = coverage / max_coverage
        base_coverage_penalty = 1.0 + (1.0 - coverage_norm) ** 2
        coverage_penalty = base_coverage_penalty ** 2  # Square it for more impact
        
        # Add 1 to CV to prevent it from dominating when near zero
        score = (e['mean_cv'] + 1.0) * consistency_penalty * count_factor * coverage_penalty
        
        combined_scores.append({
            'event_name': e['event_name'],
            'score': score,
            'mean_cv': e['mean_cv'],
            'std_across_funcs': std_across,
            'std_cv': e['std_cv'],
            'mean_count': mean_count,
            'coverage': coverage,
            'coverage_pct': (coverage / max_coverage) * 100,
            'coverage_norm': coverage_norm,
            'consistency_penalty': consistency_penalty,
            'coverage_penalty': coverage_penalty,
            'count_factor': count_factor
        })
    
    combined_scores.sort(key=lambda x: x['score'])
    
    print(f"{'Rank':<6} {'Score':<10} {'CV%':<8} {'CovNorm':<10} {'CovPen':<10} {'ConsPen':<10} {'CntFact':<10} {'Event'}")
    print("-" * 120)
    
    for i, e in enumerate(combined_scores[:15], 1):
        print(f"{i:<6} {e['score']:<10.2f} {e['mean_cv']:<8.2f} {e['coverage_norm']:<10.3f} {e['coverage_penalty']:<10.4f} {e['consistency_penalty']:<10.4f} {e['count_factor']:<10.4f} {e['event_name']}")
    
    # Calculate and display averages for ALL events
    all_events = combined_scores
    avg_score = sum(e['score'] for e in all_events) / len(all_events)
    avg_cv = sum(e['mean_cv'] for e in all_events) / len(all_events)
    avg_covnorm = sum(e['coverage_norm'] for e in all_events) / len(all_events)
    avg_covpen = sum(e['coverage_penalty'] for e in all_events) / len(all_events)
    avg_conspen = sum(e['consistency_penalty'] for e in all_events) / len(all_events)
    avg_cntfact = sum(e['count_factor'] for e in all_events) / len(all_events)
    
    print("-" * 120)
    print(f"{'AVG':<6} {avg_score:<10.2f} {avg_cv:<8.2f} {avg_covnorm:<10.3f} {avg_covpen:<10.4f} {avg_conspen:<10.4f} {avg_cntfact:<10.4f} {'(Average Across All Events)'}")
    print("=" * 120)
    
    # Save top 15
    top15_csv = os.path.join(output_dir, "top15_events_combined.csv")
    with open(top15_csv, 'w') as f:
        f.write("rank,event_name,score,mean_cv,coverage_norm,coverage_penalty,consistency_penalty,count_factor,mean_count,coverage,coverage_pct\n")
        for i, e in enumerate(combined_scores[:15], 1):
            f.write(f"{i},{e['event_name']},{e['score']:.4f},{e['mean_cv']:.4f},{e['coverage_norm']:.4f},{e['coverage_penalty']:.4f},{e['consistency_penalty']:.4f},{e['count_factor']:.4f},{e['mean_count']:.2f},{e['coverage']},{e['coverage_pct']:.2f}\n")
    print(f"\nTop 15 events saved to: {top15_csv}")
    
    # ========== Detailed breakdown for top 3 ==========
    print(f"\n{'=' * 120}")
    print("DETAILED BREAKDOWN: Top 3 Events")
    print("=" * 120)
    
    for i, e in enumerate(combined_scores[:3], 1):
        mean_count = e['mean_count']
        coverage_norm = e['coverage'] / max_coverage
        
        # Recalculate components for display
        consistency_penalty = 1.0 + (e['std_across_funcs'] / consistency_scale)
        
        if mean_count > 1:
            count_factor = 1.0 / math.log10(mean_count)
        else:
            count_factor = 10.0
        
        base_coverage_penalty = 1.0 + (1.0 - coverage_norm) ** 2
        coverage_penalty = base_coverage_penalty ** 2
        cv_plus_one = e['mean_cv'] + 1.0
        
        print(f"\n#{i}: {e['event_name']}")
        print(f"  Final Score: {e['score']:.4f}")
        print(f"  Components:")
        print(f"    • Mean CV:              {e['mean_cv']:.2f}%")
        print(f"    • CV + 1:               {cv_plus_one:.2f}  (prevents CV=0 dominance)")
        print(f"    • Consistency Penalty:  {consistency_penalty:.4f}  (1 + {e['std_across_funcs']:.2f}/{consistency_scale})")
        print(f"    • Count Factor:         {count_factor:.4f}  (1/log10({mean_count:.0f}))")
        print(f"    • Base Coverage Pen:    {base_coverage_penalty:.4f}  (1 + (1 - {coverage_norm:.3f})²)")
        print(f"    • Coverage Penalty²:    {coverage_penalty:.4f}  (squared for more impact)")
        print(f"    • Coverage:             {e['coverage']}/{max_coverage} = {e['coverage_pct']:.1f}%")
        print(f"  Calculation: ({e['mean_cv']:.2f} + 1) × {consistency_penalty:.4f} × {count_factor:.4f} × {coverage_penalty:.4f} = {e['score']:.4f}")
    
    # ========== Show worst 5 ==========
    print(f"\n{'=' * 120}")
    print("Bottom 5 Events (Highest Score - Least Stable)")
    print("=" * 120)
    for i, e in enumerate(combined_scores[-5:][::-1], 1):
        print(f"  {i}. {e['event_name']}: score={e['score']:.2f}, mean_cv={e['mean_cv']:.2f}%, coverage={e['coverage_pct']:.1f}%")

if __name__ == "__main__":
    main()
