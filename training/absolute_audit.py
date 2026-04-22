# training/absolute_audit.py
import os, sys, time, glob, base64, pickle, subprocess, requests, random
import numpy as np

# 🛠️ UTILS: Reassembly logic
def reassemble_if_needed():
    master_path = "models/asymmetric_v15_q_table.pkl"
    chunk_pattern = "models/chunks/bin_*.txt"
    if not os.path.exists(master_path):
        chunks = sorted(glob.glob(chunk_pattern))
        if chunks:
            print(f"📦 Reassembling model from {len(chunks)} chunks...")
            with open(master_path, "wb") as f_out:
                for chunk in chunks:
                    with open(chunk, "r") as f_in:
                        f_out.write(base64.b64decode(f_in.read()))
            return True
    return os.path.exists(master_path)

def run_test(name, func):
    print(f"🧪 Testing: {name}...", end=" ", flush=True)
    try:
        res = func()
        if res is True or res is None:
            print("✅ PASS")
            return True
        else:
            print(f"❌ FAIL: {res}")
            return False
    except Exception as e:
        import traceback
        print(f"💥 CRASH: {str(e)}")
        # print(traceback.format_exc())
        return False

# ================================
# 🧪 STANDALONE KERNEL FOR AUDIT
# ================================

def audit_run_episode(env, q_table, seed=42):
    obs = env.reset(seed=seed)
    done = False
    step = 0
    while not done and step < 80:
        step += 1
        actions = []
        for i in range(env.num_agents):
            state = env.get_elite_state(i)
            # Asymmetric election logic (matching agent)
            vals = [q_table.get((state, a), 0.0) for a in [0,1,2,3,4,5,6]]
            max_val = max(vals)
            best_actions = [idx for idx, v in enumerate(vals) if v == max_val]
            action = best_actions[i % len(best_actions)]
            actions.append(action)
        obs, _, done, info = env.step(actions)
    return info

# ================================
# 🧪 TEST FUNCTIONS
# ================================

def test_reassembly():
    return reassemble_if_needed()

def test_server_boot():
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = os.getcwd()
    # Use explicit venv if possible, otherwise system
    py_path = sys.executable 
    p = subprocess.Popen([py_path, "server/app.py"], env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    try:
        resp = requests.get("http://127.0.0.1:7860/health", timeout=5)
        p.terminate()
        return resp.status_code == 200
    except Exception as e:
        # print(f"Error: {e}")
        p.terminate()
        return False

def test_api_endpoints():
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = os.getcwd()
    p = subprocess.Popen([sys.executable, "server/app.py"], env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    try:
        results = []
        results.append(requests.get("http://127.0.0.1:7860/", timeout=5).status_code == 200)
        results.append(requests.post("http://127.0.0.1:7860/reset", json={"task":"easy"}, timeout=5).status_code == 200)
        results.append(requests.post("http://127.0.0.1:7860/step", json={"actions":[0, 1]}, timeout=5).status_code == 200)
        results.append(requests.post("http://127.0.0.1:7860/grader", json={}, timeout=5).status_code == 200)
        results.append(requests.post("http://127.0.0.1:7860/reason", params={"agent_idx":0}, timeout=5).status_code == 200)
        p.terminate()
        return all(results)
    except:
        p.terminate()
        return False

def test_state_consistency():
    sys.path.insert(0, os.getcwd())
    from env.core.environment import GridEnv
    from env.tasks.hard import get_task
    env = GridEnv(get_task())
    env.reset(seed=42) # 🛡️ Required to populate agent positions
    state = env.get_elite_state(0)
    if not isinstance(state, (tuple, list)): return f"State is not iterable: {type(state)}"
    if len(state) != 12: return f"Expected 12 elements, got {len(state)}: {state}"
    return True

def test_q_table_integrity():
    with open("models/asymmetric_v15_q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    if len(q_table) < 5000: return f"Q-table too small: {len(q_table)}"
    vals = list(q_table.values())
    non_zero = [v for v in vals if v != 0]
    if len(non_zero) < 100: return "Low knowledge distribution in Q-table"
    return True

def test_inference_performance():
    from env.core.environment import GridEnv
    from env.tasks.easy import get_task
    with open("models/asymmetric_v15_q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    env = GridEnv(get_task())
    info = audit_run_episode(env, q_table, seed=42)
    if info['total_deliveries'] < 1: return f"System failed to deliver even 1 item (0). Seed 42."
    return True

def test_coordination_diversity():
    from env.core.environment import GridEnv
    from env.tasks.hard import get_task
    with open("models/asymmetric_v15_q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    env = GridEnv(get_task())
    env.reset(seed=42)
    
    sync_count = 0
    for _ in range(50):
        actions = []
        for i in range(2):
            state = env.get_elite_state(i)
            vals = [q_table.get((state, a), 0.0) for a in [0,1,2,3,4,5,6]]
            max_val = max(vals)
            best_actions = [idx for idx, v in enumerate(vals) if v == max_val]
            action = best_actions[i % len(best_actions)]
            actions.append(action)
        if actions[0] == actions[1]: sync_count += 1
        _, _, done, _ = env.step(actions)
        if done: break
    
    if sync_count > 45: return f"Extreme action synchronization (>90%): {sync_count}/50"
    return True

def test_seed_consistency():
    from env.core.environment import GridEnv
    from env.tasks.easy import get_task
    with open("models/asymmetric_v15_q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
        
    env1 = GridEnv(get_task())
    info1 = audit_run_episode(env1, q_table, seed=99)
    env2 = GridEnv(get_task())
    info2 = audit_run_episode(env2, q_table, seed=99)
    
    if info1['total_deliveries'] != info2['total_deliveries'] or info1['step_count'] != info2['step_count']:
        return f"Non-deterministic: {info1} vs {info2}"
    return True

def test_fault_tolerance():
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = os.getcwd()
    p = subprocess.Popen([sys.executable, "server/app.py"], env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    try:
        requests.post("http://127.0.0.1:7860/reset")
        resp = requests.post("http://127.0.0.1:7860/step", json={"actions":[999, -1]})
        p.terminate()
        return resp.status_code in [200, 400, 422]
    except:
        p.terminate()
        return False

# ================================
# 🚀 MAIN AUDIT EXECUTION
# ================================
if __name__ == "__main__":
    print("\n🚩 AIDK ABSOLUTE FINAL AUDIT (V15) 🚩")
    print("="*40)
    
    tests = [
        ("Model Reassembly", test_reassembly),
        ("Server Boot Stability", test_server_boot),
        ("API Endpoint Health", test_api_endpoints),
        ("State Consistency (12-Element)", test_state_consistency),
        ("Q-Table Integrity", test_q_table_integrity),
        ("Inference Performance (Easy)", test_inference_performance),
        ("Swarm Diversity (Non-Sync)", test_coordination_diversity),
        ("Seed Determinism", test_seed_consistency),
        ("Fault Tolerance (Garbage Input)", test_fault_tolerance),
    ]
    
    results = []
    for name, func in tests:
        results.append(run_test(name, func))
    
    passed = sum(results)
    print("="*40)
    print(f"🏆 AUDIT COMPLETE: {passed}/{len(tests)} PASSED")
    
    if passed == len(tests):
        print("🚀 SYSTEM IS TECHNICALLY READY FOR SUBMISSION.")
        sys.exit(0)
    else:
        print("🚨 CRITICAL FAILURES DETECTED. FIX BEFORE SUBMITTING.")
        sys.exit(1)
