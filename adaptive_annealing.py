import numpy as np
import math
import random
import argparse
import sys
import signal
from tqdm import tqdm
import multiprocessing
import time
import os
import itertools
from multiprocessing import Value

# 導入自適應替代模型和原始驗證器
from adaptive_surrogate import compute_m_height as adaptive_compute_m_height
from verifier import compute_m_height as exact_compute_m_height

# 全局標誌用於Ctrl+C
user_interrupted = False

# 工作進程全局變量
worker_cost_counter = None
current_strategy = None
current_sample_rate = None

def init_worker(counter, strategy, sample_rate):
    """初始化工作進程的全局變量"""
    global worker_cost_counter, current_strategy, current_sample_rate
    worker_cost_counter = counter
    current_strategy = strategy
    current_sample_rate = sample_rate

def signal_handler(sig, frame):
    """處理Ctrl+C信號"""
    global user_interrupted
    if user_interrupted:
        print('\nCtrl+C再次檢測到，強制退出。')
        sys.exit(1)
    print('\nCtrl+C檢測到，請求工作進程完成當前步驟後停止...')
    user_interrupted = True

def calculate_cost(G, m):
    """使用自適應替代模型計算m-height"""
    global worker_cost_counter, current_strategy, current_sample_rate
    
    try:
        # 使用自適應替代模型，策略和採樣率由全局變量控制
        height = adaptive_compute_m_height(
            G, m, 
            strategy=current_strategy, 
            sample_rate=current_sample_rate
        )
        return height
    except Exception as e:
        # print(f"計算錯誤: {e}")
        return float('inf')

def get_neighbor(P, element_range):
    """生成鄰居P矩陣，改變一個元素"""
    k, p_cols = P.shape
    P_new = P.copy()
    max_retries_neighbor = 10
    max_retries_value = 20

    for _ in range(max_retries_neighbor):
        row_idx = random.randrange(k)
        col_idx = random.randrange(p_cols)
        current_val = P_new[row_idx, col_idx]

        possible_values = list(range(element_range[0], element_range[1] + 1))
        if len(possible_values) <= 1:
            continue

        new_val = current_val
        attempt_val = 0
        while new_val == current_val and attempt_val < max_retries_value:
             new_val = random.choice(possible_values)
             attempt_val += 1

        if new_val == current_val:
             continue

        original_value_in_col = P_new[row_idx, col_idx]
        P_new[row_idx, col_idx] = new_val

        if not np.all(P_new[:, col_idx] == 0):
            return P_new
        else:
            P_new[row_idx, col_idx] = original_value_in_col

    return P

def simulated_annealing_core(initial_P, k, n, m, T_max, T_min, alpha, iter_per_temp, element_range, worker_id=0, seed=None):
    """退火算法核心邏輯"""
    global worker_cost_counter

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if not (1 <= m <= n - 1):
        return None, float('inf')

    I_k = np.eye(k, dtype=initial_P.dtype)

    current_P = initial_P
    current_G = np.hstack((I_k, current_P))
    current_cost = calculate_cost(current_G, m)

    if worker_cost_counter is not None:
        with worker_cost_counter.get_lock():
            worker_cost_counter.value += 1

    best_P = current_P
    best_cost = current_cost

    T = T_max
    temp_step = 0

    while T > T_min:
        for i in range(iter_per_temp):
            neighbor_P = get_neighbor(current_P, element_range)
            neighbor_G = np.hstack((I_k, neighbor_P))
            neighbor_cost = calculate_cost(neighbor_G, m)

            if worker_cost_counter is not None:
                with worker_cost_counter.get_lock():
                    worker_cost_counter.value += 1

            delta_E = neighbor_cost - current_cost

            # 接受邏輯
            accept = False
            if neighbor_cost == float('inf') and current_cost == float('inf'):
                accept = False
            elif neighbor_cost == float('inf'):
                accept = False
            elif current_cost == float('inf'):
                accept = True
            elif delta_E < 0:
                accept = True
            else:
                if delta_E == 0:
                    acceptance_prob = 1.0
                elif T <= 0:
                     acceptance_prob = 0.0
                else:
                    exponent = -delta_E / T
                    if exponent > 700:
                        acceptance_prob = 0.0
                    else:
                        acceptance_prob = math.exp(exponent)

                if random.random() < acceptance_prob:
                    accept = True

            if accept:
                current_P = neighbor_P
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_P = current_P
                    best_cost = current_cost

        T *= alpha
        temp_step += 1

    best_G = np.hstack((I_k, best_P))
    return best_G, best_cost

def is_valid_P(P):
    """檢查P矩陣的每一列是否都不是全零向量"""
    k, p_cols = P.shape
    for j in range(p_cols):
        if np.all(P[:, j] == 0):
            return False
    return True

def run_single_annealing_instance(args_tuple):
    """運行單個退火實例的工作函數"""
    worker_id, k, n, m, element_range, T_max, T_min, alpha, iter_per_temp, seed = args_tuple
    
    if seed is None:
        base_seed = int(time.time() * 1000) + random.randint(0, 10000) + worker_id + os.getpid()
        seed = base_seed % (2**32)
    random.seed(seed)
    np.random.seed(seed)

    p_cols = n - k
    initial_P = None
    max_init_retries = 100
    element_min, element_max = element_range

    for attempt in range(max_init_retries):
        P_candidate = np.random.randint(element_min, element_max + 1, size=(k, p_cols))
        if is_valid_P(P_candidate):
            initial_P = P_candidate
            break

    if initial_P is None:
        return None, float('inf'), worker_id

    best_generator, min_height = simulated_annealing_core(
        initial_P=initial_P,
        k=k,
        n=n,
        m=m,
        T_max=T_max,
        T_min=T_min,
        alpha=alpha,
        iter_per_temp=iter_per_temp,
        element_range=element_range,
        worker_id=worker_id,
        seed=seed
    )

    return best_generator, min_height, worker_id

def verify_final_result(G, m, verbose=False):
    """使用原始精確方法驗證最終結果"""
    if verbose:
        print("使用原始精確方法驗證最終結果...")
    
    start_time = time.time()
    try:
        exact_height = exact_compute_m_height(G, m)
        verify_time = time.time() - start_time
        
        if verbose:
            print(f"驗證完成，用時 {verify_time:.2f}秒")
            print(f"精確m-height: {exact_height:.6g}")
        
        return exact_height
    except Exception as e:
        if verbose:
            print(f"驗證失敗: {e}")
        return float('inf')

def recommend_strategy(k, n, m):
    """根據問題大小推薦最佳策略和採樣率"""
    complexity = k * n * m
    
    if complexity < 50:  # 小問題
        return "adaptive", 0.2
    elif complexity < 100:  # 中等問題
        return "sampled", 0.1
    else:  # 大問題
        return "sampled", 0.05

if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="使用自適應替代模型進行退火搜索，尋找最小化m-height的系統生成矩陣")

    # 問題參數
    parser.add_argument('-k', type=int, default=5, help='信息比特數 (行數). 默認: 5')
    parser.add_argument('-n', type=int, default=9, help='碼字比特數 (列數). 默認: 9')
    parser.add_argument('-m', type=int, default=2, help='m-height參數 (須滿足 1 <= m <= n-1). 默認: 2')
    parser.add_argument('--element-min', type=int, default=-100, help='P矩陣元素最小值. 默認: -100')
    parser.add_argument('--element-max', type=int, default=100, help='P矩陣元素最大值. 默認: 100')
    
    # 退火參數
    parser.add_argument('--t-max', type=float, default=50.0, help='初始溫度. 默認: 50.0')
    parser.add_argument('--t-min', type=float, default=1.0, help='最終溫度. 默認: 1.0')
    parser.add_argument('--alpha', type=float, default=0.95, help='冷卻率. 默認: 0.95')
    parser.add_argument('--iter-per-temp', type=int, default=5, help='每個溫度級別的迭代次數. 默認: 5')
    
    # 自適應模型參數
    parser.add_argument('--strategy', type=str, choices=['heuristic', 'sampled', 'adaptive', 'auto'], 
                      default='auto', help='替代模型策略. 默認: auto (自動選擇)')
    parser.add_argument('--sample-rate', type=float, default=0.05, 
                      help='採樣率 (僅用於sampled策略). 默認: 0.05')
    parser.add_argument('--verify-final', action='store_true', 
                      help='使用精確方法驗證最終結果')
    
    # 並行參數
    try:
        default_workers = multiprocessing.cpu_count()
    except NotImplementedError:
        default_workers = 1
        print("警告: 無法確定CPU核心數，默認使用1個工作進程")
    parser.add_argument('--workers', type=int, default=default_workers,
                        help=f'並行退火運行的數量. 默認: {default_workers} (邏輯CPU核心數)')

    args = parser.parse_args()

    # 提取參數
    K = args.k
    N = args.n
    M = args.m
    ELEMENT_MIN = args.element_min
    ELEMENT_MAX = args.element_max
    ELEMENT_RANGE = (ELEMENT_MIN, ELEMENT_MAX)
    T_MAX = args.t_max
    T_MIN = args.t_min
    ALPHA = args.alpha
    ITER_PER_TEMP = args.iter_per_temp
    NUM_WORKERS = args.workers
    VERIFY_FINAL = args.verify_final
    
    # 確定策略和採樣率
    if args.strategy == 'auto':
        STRATEGY, SAMPLE_RATE = recommend_strategy(K, N, M)
        print(f"自動選擇策略: {STRATEGY}, 採樣率: {SAMPLE_RATE}")
    else:
        STRATEGY = args.strategy
        SAMPLE_RATE = args.sample_rate
    
    # 驗證參數
    if K < 1:
        parser.error(f"k ({K}) 必須至少為1")
    if N <= K:
        parser.error(f"n ({N}) 必須大於 k ({K}) 才能是有效的系統碼")
    p_cols = N - K
    if p_cols < 1:
        parser.error(f"n-k ({p_cols}) 必須至少為1，請檢查n和k的值")
    if not (1 <= M <= N - 1):
        parser.error(f"m ({M}) 必須在1和n-1 ({N-1})之間")
    if ELEMENT_MIN > ELEMENT_MAX:
        parser.error(f"element-min ({ELEMENT_MIN}) 不能大於 element-max ({ELEMENT_MAX})")
    if ELEMENT_MIN == 0 and ELEMENT_MAX == 0:
        parser.error(f"元素範圍為[0, 0]，無法滿足非零列約束")
    if NUM_WORKERS < 1:
        parser.error(f"工作進程數量 ({NUM_WORKERS}) 必須至少為1")
    
    # 計算總共的步驟數，用於進度條
    total_cost_calls = 0
    if T_MAX > T_MIN and 0 < ALPHA < 1:
        try:
            num_steps = math.ceil(math.log(T_MIN / T_MAX) / math.log(ALPHA))
            total_cost_calls = NUM_WORKERS * num_steps * ITER_PER_TEMP
            print(f"每個工作進程預計有約 {num_steps} 個溫度步驟")
            print(f"在 {NUM_WORKERS} 個工作進程中預計總計算次數: {total_cost_calls}")
        except ValueError:
            print("警告: 無法計算總步驟數，進度條可能不準確")
            total_cost_calls = 1
    else:
        print("警告: T_max <= T_min 或 alpha 值不在(0,1)範圍內，進度條可能不準確")
        total_cost_calls = 1
    
    # 創建共享計數器
    cost_call_counter = Value('i', 0)
    
    print(f"啟動並行模擬退火，使用 {NUM_WORKERS} 個工作進程")
    print(f"目標: 最小化 ({N},{K},{M}) 碼的m-height")
    print(f"P矩陣元素範圍: [{ELEMENT_RANGE[0]}, {ELEMENT_RANGE[1]}]")
    print(f"退火參數: T_max={T_MAX}, T_min={T_MIN}, alpha={ALPHA}, iter_per_temp={ITER_PER_TEMP}")
    print(f"使用自適應替代模型，策略: {STRATEGY}, 採樣率: {SAMPLE_RATE}")
    print("按Ctrl+C可提前停止 (可能需要一刻鐘讓所有工作進程終止)")
    
    # 為每個工作進程準備參數
    worker_args = [
        (i, K, N, M, ELEMENT_RANGE, T_MAX, T_MIN, ALPHA, ITER_PER_TEMP, None)
        for i in range(NUM_WORKERS)
    ]
    
    # 設置信號處理器
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)
    
    pool = None
    results = []
    
    try:
        with multiprocessing.Pool(
                processes=NUM_WORKERS,
                initializer=init_worker,
                initargs=(cost_call_counter, STRATEGY, SAMPLE_RATE)
            ) as pool:
                
            result_iterator = pool.imap_unordered(run_single_annealing_instance, worker_args)
            
            print(f"啟動 {NUM_WORKERS} 個退火搜索進程...")
            processed_count = 0
            
            with tqdm(total=total_cost_calls, desc="計算進度", unit="次") as pbar:
                while processed_count < NUM_WORKERS:
                    if user_interrupted:
                        print("\n檢測到中斷信號，退出結果循環...")
                        break
                    
                    try:
                        result = result_iterator.next(timeout=0.5)
                        
                        best_G, cost, worker_id = result
                        if best_G is not None:
                            results.append((best_G, cost))
                            # 更新進度條描述來顯示當前最佳解
                            pbar.set_description(f"計算進度 (當前最佳: {min([c for _, c in results] + [float('inf')]):.4f})")
                        else:
                            print(f"工作進程 {worker_id} 未能產生有效結果")
                        
                        processed_count += 1
                        pbar.update(0)  # 更新顯示但不增加計數
                        
                    except StopIteration:
                        break
                    except multiprocessing.TimeoutError:
                        # 超時，更新進度條但繼續等待
                        pbar.n = cost_call_counter.value
                        pbar.refresh()
                        continue
                    except Exception as worker_exc:
                        print(f"\n處理工作進程結果時發生錯誤: {worker_exc}")
                        processed_count += 1
                        pbar.update(1)
                    
                    # 更新進度條使用共享計數器
                    pbar.n = cost_call_counter.value
                    pbar.refresh()
            
            if user_interrupted:
                print("\n終止運行中的工作進程...")
                pool.terminate()
                pool.join()
                print("工作進程池已終止")
            else:
                print("\n所有工作進程已完成或失敗")
                
    except KeyboardInterrupt:
        print("\n主進程捕獲鍵盤中斷信號")
        sys.exit(1)
    except Exception as e:
        print(f"\n主進程發生意外錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
    
    # 彙總結果
    if not results:
        print("\n沒有從任何工作進程獲得有效結果")
        if user_interrupted:
            print("退火過程被中斷")
        sys.exit(1)
    
    # 按照cost排序結果找出最佳解
    results.sort(key=lambda x: x[1])
    overall_best_G, overall_best_cost = results[0]
    
    print("\n--- 最佳結果 ---")
    print(f"從 {len(results)} 個成功的工作進程中獲得")
    print("最佳系統生成矩陣 G = [I|P]:")
    with np.printoptions(precision=3, suppress=True):
        print(overall_best_G)
    print(f"最小m-height (m={M}): {overall_best_cost:.6g}")
    
    # 使用精確方法驗證最終結果
    if VERIFY_FINAL:
        print("\n--- 最終驗證 ---")
        exact_cost = verify_final_result(overall_best_G, M, verbose=True)
        
        if exact_cost != overall_best_cost:
            print(f"注意: 替代模型估計 ({overall_best_cost:.6g}) 與精確驗證 ({exact_cost:.6g}) 不同")
            if exact_cost == float('inf') and overall_best_cost != float('inf'):
                print("警告: 替代模型未能檢測出無界情況!")
            elif exact_cost != float('inf') and overall_best_cost == float('inf'):
                print("注意: 替代模型過於保守 (無界情況的誤報)")
            else:
                rel_error = abs(exact_cost - overall_best_cost) / (exact_cost if exact_cost != 0 else 1)
                print(f"替代模型相對誤差: {rel_error:.2%}")
    
    if user_interrupted:
        print("\n過程被中斷。報告的最佳結果基於中斷前完成的運行") 