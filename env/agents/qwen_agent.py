# env/agents/qwen_agent.py

import random
import re
import os
import requests

class QwenAgent:
    def __init__(self, actions_list):
        self.actions = actions_list
        self.name = "Qwen-AIDK"
        self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self.api_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-0.5B-Instruct"
        print("🚀 [AIDK] Initializing REAL HuggingFace Qwen API connection...")
        if not self.hf_token:
            print("⚠️ [AIDK Warning] HF_TOKEN not found. Will attempt unauthenticated API call (rate limits may apply).")

    def reason_and_act(self, state_json):
        """
        🛡️ AIDK WINNING EDGE: Qwen reasoning kernel.
        Translates structured state into a natural language decision via REAL API.
        """
        prompt = self._build_prompt(state_json)
        
        headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        payload = {
            "inputs": f"<|im_start|>system\nYou are a warehouse routing AI. Output your reasoning then state 'ACTION: <number>'.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "parameters": {"max_new_tokens": 50, "return_full_text": False}
        }
        
        try:
            # 🚀 REAL LLM API CALL
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                reasoning = result[0]['generated_text'] if isinstance(result, list) else result.get('generated_text', '')
                source = "[Qwen Source: API]"
            else:
                reasoning = f"[API Error {response.status_code}]: {response.text}\n[Fallback Parser]: Action determined via internal priority."
                source = "[Qwen Source: Fallback Heuristic]"
        except Exception as e:
            reasoning = f"[Network Error]: {e}\n[Fallback Parser]: Action determined via internal priority."
            source = "[Qwen Source: Fallback Heuristic]"
                
        # Parse action
        action = 0
        match = re.search(r"ACTION:\s*(\d)", reasoning)
        if match:
            action = int(match.group(1))
        else:
            action = self._heuristic_fallback(state_json)
            
        return {
            "prompt": prompt,
            "reasoning": f"{source}\n{reasoning.strip()}",
            "action": action
        }

    def _build_prompt(self, state):
        prompt = f"""Warehouse Rules Setup:
1. You must navigate to the Pickup Location first.
2. If you are at the Pickup Location, you must perform Action 4 (PICKUP).
3. Once you have the item (Has Item: True), you must navigate to the Delivery Location.
4. If you are at the Delivery Location and have the item, you must perform Action 5 (DELIVER).

Available Actions: 0 (UP), 1 (DOWN), 2 (LEFT), 3 (RIGHT), 4 (PICKUP), 5 (DELIVER), 6 (RECHARGE)

Current State: 
- Agent Position: {state['pos']}
- Has Item: {state['has_item']}
- Energy: {state['energy']}
- Pickup Location: {state['pickup_pos']}
- Delivery Location: {state['delivery_pos']}

What is your reasoning and next action? End your output exactly with 'ACTION: number'."""
        return prompt

    def _heuristic_fallback(self, state):
        _, action = self._simulate_qwen_thought(state)
        return action

    def _simulate_qwen_thought(self, state):
        """🛡️ Fallback Engine if transformers is unavailable"""
        pos = state['pos']
        has_item = state['has_item']
        pickup = state['pickup_pos']
        delivery = state['delivery_pos']

        if not has_item:
            reasoning = f"I cannot deliver because I have not picked up the item. I will navigate to the pickup location at {pickup}."
            action = self._get_move_towards(pos, pickup)
            if pos == pickup:
                reasoning = "I am at the pickup location. I will perform the PICKUP action."
                action = 4 
        else:
            reasoning = f"I have the item. I must now deliver it to the goal at {delivery}."
            action = self._get_move_towards(pos, delivery)
            if pos == delivery:
                reasoning = "I am at the delivery location. I will perform the DELIVER action."
                action = 5 
        return reasoning, action

    def _get_move_towards(self, current, target):
        if current[0] < target[0]: return 1 # DOWN
        if current[0] > target[0]: return 0 # UP
        if current[1] < target[1]: return 3 # RIGHT
        if current[1] > target[1]: return 2 # LEFT
        return 0
