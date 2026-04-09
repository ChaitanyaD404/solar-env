import os
from openai import OpenAI
from env import SolarEnv

# -------------------------
# ENV VARIABLES
# -------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("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")

# -------------------------
# CLIENT SETUP
# -------------------------
client = None
if API_KEY:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# -------------------------
# INIT ENV
# -------------------------
env = SolarEnv()
obs = env.reset("easy")

rewards = []
done = False
step_count = 0
success = True

print(f"[START] task=easy env=solar-env model={MODEL_NAME}")

try:
    while not done and step_count < 24:
        step_count += 1

        # -------------------------
        # ACTION SELECTION
        # -------------------------
        if client:
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": str(obs)}],
                    temperature=0
                )
                action = response.choices[0].message.content.strip()
            except Exception:
                action = "noop"
        else:
            # 🔥 FALLBACK (NO API CASE)
            if obs["sun"] > 0.7:
                action = "charge"
            elif obs["battery"] > 70:
                action = "discharge"
            elif obs["dirt"][0] > 0.5:
                action = "clean_1"
            else:
                action = "noop"

        # -------------------------
        # STEP ENV
        # -------------------------
        try:
            obs, reward, done, info = env.step(action)
            reward_value = round(reward.score, 2)
            error = "null"
        except Exception as e:
            reward_value = 0.00
            error = str(e)
            done = True
            success = False

        rewards.append(reward_value)

        print(f"[STEP] step={step_count} action={action} reward={reward_value:.2f} done={str(done).lower()} error={error}")

except Exception as e:
    success = False
    print(f"[STEP] step={step_count} action=error reward=0.00 done=true error={str(e)}")

# -------------------------
# FINAL SCORE
# -------------------------
if len(rewards) > 0:
    score = round(sum(rewards) / (len(rewards) * 3), 3)  # normalize
else:
    score = 0.0

reward_str = ",".join([f"{r:.2f}" for r in rewards])

print(f"[END] success={str(success).lower()} steps={step_count} score={score:.3f} rewards={reward_str}")
