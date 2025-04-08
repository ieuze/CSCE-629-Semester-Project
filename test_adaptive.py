import numpy as np
import time
import matplotlib.pyplot as plt
from verifier import compute_m_height as original_compute_m_height
from adaptive_surrogate import compute_m_height as adaptive_compute_m_height
from surrogate import compute_m_height as surrogate_compute_m_height

def test_comparison():
    """比較三種方法的精度和速度: 原始驗證器、基本替代模型和自適應替代模型"""
    # 測試案例
    test_cases = [
        # 小型矩陣
        {
            "name": "Small Matrix",
            "G": np.array([[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 2]]),
            "m_values": [1, 2, 3]
        },
        # 中型矩陣
        {
            "name": "Medium Matrix",
            "G": np.random.randint(-5, 6, size=(4, 8)),
            "m_values": [1, 3, 5]
        },
        # 大型矩陣 (9,5,2) 案例
        {
            "name": "(9,5,2) Case",
            "G": np.hstack((np.eye(5), np.random.randint(-20, 21, size=(5, 4)))),
            "m_values": [2]
        }
    ]
    
    # 存儲結果
    results = []
    
    print("比較三種m-height計算方法:")
    print("=" * 60)
    
    for case in test_cases:
        print(f"\n測試: {case['name']}")
        G = case["G"]
        k, n = G.shape
        print(f"矩陣形狀: ({k}x{n})")
        
        for m in case["m_values"]:
            print(f"\n  m={m}:")
            
            # 使用原始方法
            try:
                start_time = time.time()
                original_height = original_compute_m_height(G, m)
                original_time = time.time() - start_time
                print(f"  原始方法:    {original_height:.6g} (用時 {original_time:.4f}s)")
            except Exception as e:
                original_height = float('inf')
                original_time = time.time() - start_time
                print(f"  原始方法:    錯誤: {e} (用時 {original_time:.4f}s)")
            
            # 使用基本替代模型
            try:
                start_time = time.time()
                surrogate_height = surrogate_compute_m_height(G, m)
                surrogate_time = time.time() - start_time
                print(f"  基本替代:    {surrogate_height:.6g} (用時 {surrogate_time:.4f}s, 加速 {original_time/surrogate_time:.1f}x)")
                
                if np.isfinite(original_height) and np.isfinite(surrogate_height):
                    rel_error = abs(surrogate_height - original_height) / max(1e-10, original_height)
                    print(f"    相對誤差: {rel_error:.2%}")
                elif original_height == surrogate_height:
                    print("    結果相同")
                else:
                    print("    結果不同")
            except Exception as e:
                surrogate_height = float('inf')
                surrogate_time = time.time() - start_time
                print(f"  基本替代:    錯誤: {e} (用時 {surrogate_time:.4f}s)")
            
            # 使用自適應策略
            strategies = ["heuristic", "sampled", "adaptive"]
            adaptive_results = []
            
            for strategy in strategies:
                try:
                    start_time = time.time()
                    adaptive_height = adaptive_compute_m_height(G, m, strategy=strategy)
                    adaptive_time = time.time() - start_time
                    
                    if np.isfinite(original_height) and np.isfinite(adaptive_height):
                        rel_error = abs(adaptive_height - original_height) / max(1e-10, original_height)
                        error_str = f", 誤差 {rel_error:.2%}"
                    elif original_height == adaptive_height:
                        error_str = ", 結果相同"
                    else:
                        error_str = ", 結果不同"
                    
                    print(f"  自適應({strategy}): {adaptive_height:.6g} (用時 {adaptive_time:.4f}s, 加速 {original_time/adaptive_time:.1f}x{error_str})")
                    
                    adaptive_results.append({
                        "strategy": strategy,
                        "height": adaptive_height,
                        "time": adaptive_time,
                        "speedup": original_time / adaptive_time if adaptive_time > 0 else float('inf'),
                        "error": abs(adaptive_height - original_height) / max(1e-10, original_height) 
                              if (np.isfinite(original_height) and np.isfinite(adaptive_height)) else float('nan')
                    })
                    
                except Exception as e:
                    adaptive_time = time.time() - start_time
                    print(f"  自適應({strategy}): 錯誤: {e} (用時 {adaptive_time:.4f}s)")
            
            # 存儲結果
            results.append({
                "case": case["name"],
                "m": m,
                "matrix_shape": G.shape,
                "original": {"height": original_height, "time": original_time},
                "surrogate": {"height": surrogate_height, "time": surrogate_time},
                "adaptive": adaptive_results
            })
    
    # 繪製結果圖表
    plot_comparison_results(results)
    
    return results

def plot_comparison_results(results):
    """繪製結果比較圖表"""
    try:
        # 分離數據
        speedups = {"heuristic": [], "sampled": [], "adaptive": [], "surrogate": []}
        errors = {"heuristic": [], "sampled": [], "adaptive": [], "surrogate": []}
        
        for r in results:
            # 處理基本替代模型
            orig_time = r["original"]["time"]
            surr_time = r["surrogate"]["time"]
            
            if surr_time > 0:
                speedups["surrogate"].append(orig_time / surr_time)
            
            if (np.isfinite(r["original"]["height"]) and 
                np.isfinite(r["surrogate"]["height"])):
                errors["surrogate"].append(
                    abs(r["surrogate"]["height"] - r["original"]["height"]) / 
                    max(1e-10, r["original"]["height"])
                )
            
            # 處理自適應策略
            for a in r["adaptive"]:
                if a["time"] > 0:
                    speedups[a["strategy"]].append(a["speedup"])
                
                if not np.isnan(a["error"]):
                    errors[a["strategy"]].append(a["error"])
        
        # 創建速度對比圖
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        labels = []
        values = []
        for strategy, data in speedups.items():
            if data:  # 確保存在數據
                labels.append(strategy)
                values.append(np.median(data))
        
        bars = plt.bar(labels, values)
        plt.yscale('log')
        plt.ylabel('中位加速比 (log scale)')
        plt.title('速度對比 (越高越好)')
        plt.grid(True, alpha=0.3)
        
        # 在柱子上方標註具體數值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}x',
                    ha='center', va='bottom', rotation=0)
        
        # 創建誤差對比圖
        plt.subplot(1, 2, 2)
        labels = []
        values = []
        for strategy, data in errors.items():
            if data:  # 確保存在數據
                labels.append(strategy)
                values.append(np.median(data) * 100)  # 轉換為百分比
        
        bars = plt.bar(labels, values)
        plt.ylabel('中位相對誤差 (%)')
        plt.title('精確度對比 (越低越好)')
        plt.grid(True, alpha=0.3)
        
        # 在柱子上方標註具體數值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig('method_comparison.png')
        print("\n結果圖表已保存為 'method_comparison.png'")
        
    except Exception as e:
        print(f"繪圖錯誤: {e}")

if __name__ == "__main__":
    try:
        test_comparison()
    except ImportError as e:
        if "matplotlib" in str(e):
            print("缺少matplotlib，跳過繪圖")
        else:
            print(f"錯誤: {e}")
    except Exception as e:
        print(f"測試過程中出錯: {e}") 