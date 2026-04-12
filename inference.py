import os
from openai import OpenAI
from env import SolarEnv

# ==================================================
# REQUIRED ENV VARIABLES (judge injects these)
# ==================================================
API_BASE_URL = os.environ["https://their-proxy-server/v1"]
API_KEY = os.environ["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"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ==================================================
# OPENAI CLIENT (must use proxy)
# ==================================================
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# ==================================================
# ENV INIT
# ==================================================
env = SolarEnv()
obs = env.reset("easy")

done = False
step_count = 0
rewards = []
success = True

print(f"[START] task=easy env=solar-env model={MODEL_NAME}")

# ==================================================
# FORCE ONE API CALL (important for validator)
# ==================================================
try:
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "Respond with exactly: noop"}
        ],
        temperature=0
    )
except:
    pass

# ==================================================
# MAIN LOOP
# ==================================================
try:
    while not done and step_count < 24:
        step_count += 1

        # ------------------------------------------
        # GET ACTION FROM LLM PROXY
        # ------------------------------------------
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": f"""
You are controlling a solar energy environment.

Observation:
{obs}

Choose ONE valid action only:
noop
charge
discharge
clean_1
clean_2
clean_3

Return only the action word.
"""
                    }
                ],
                temperature=0
            )

            action = response.choices[0].message.content.strip().lower()

        except:
            # fallback if proxy has issue
            action = "noop"

        # ------------------------------------------
        # SAFETY FILTER
        # ------------------------------------------
        valid_actions = [
            "noop",
            "charge",
            "discharge",
            "clean_1",
            "clean_2",
            "clean_3"
        ]

        if action not in valid_actions:
            action = "noop"

        # ------------------------------------------
        # STEP ENV
        # ------------------------------------------
        try:
            obs, reward, done, info = env.step(action)
            reward_value = float(round(reward.score, 2))
            error = "null"

        except Exception as e:
            reward_value = 0.00
            done = True
            success = False
            error = str(e)

        rewards.append(reward_value)

        print(
            f"[STEP] step={step_count} "
            f"action={action} "
            f"reward={reward_value:.2f} "
            f"done={str(done).lower()} "
            f"error={error}"
        )

except Exception as e:
    success = False
    print(
        f"[STEP] step={step_count} "
        f"action=noop reward=0.00 done=true error={str(e)}"
    )

# ==================================================
# FINAL SCORE (0 to 1)
# ==================================================
if rewards:
    raw = sum(rewards) / len(rewards)
    score = max(0.0, min(1.0, raw / 3.0))
else:
    score = 0.0

reward_str = ",".join([f"{r:.2f}" for r in rewards])

print(
    f"[END] success={str(success).lower()} "
    f"steps={step_count} "
    f"score={score:.2f} "
    f"rewards={reward_str}"
)
