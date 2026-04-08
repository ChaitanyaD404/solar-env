from models import Observation, Action, Reward

class SolarEnv:
    def __init__(self):
        self.state_data = None

    def reset(self, task="easy"):
        if task == "easy":
            dirt = [0.1, 0.1, 0.1]
            battery = 50
        elif task == "medium":
            dirt = [0.3, 0.3, 0.3]
            battery = 40
        else:  # hard
            dirt = [0.5, 0.5, 0.5]
            battery = 30

        self.state_data = {
            "dirt": dirt,
            "sun": 0.8,
            "battery": battery,
            "hour": 0
        }

        return Observation(**self.state_data)

    def step(self, action: Action):
        reward = 0.0

        dirt = self.state_data["dirt"]
        sun = self.state_data["sun"]

        # 🔆 Power generation
        power = sum([(1 - d) * sun for d in dirt])

        # 🎮 Actions
        if action.action == "clean_1":
            dirt[0] = max(0, dirt[0] - 0.5)
            reward -= 0.1

        elif action.action == "clean_2":
            dirt[1] = max(0, dirt[1] - 0.5)
            reward -= 0.1

        elif action.action == "clean_3":
            dirt[2] = max(0, dirt[2] - 0.5)
            reward -= 0.1

        elif action.action == "charge":
            self.state_data["battery"] += 5
            reward -= 0.05

        elif action.action == "discharge":
            self.state_data["battery"] -= 5
            reward += 0.1

        elif action.action == "noop":
            pass

        # 🎁 Reward from energy
        reward += power

        # ⚠️ Penalty for too much dirt
        if sum(dirt) > 1.5:
            reward -= 0.2

        # ⚠️ Battery penalty
        if self.state_data["battery"] > 100 or self.state_data["battery"] < 0:
            reward -= 0.3

        # ⏱ Update time
        self.state_data["hour"] += 1
        hour = self.state_data["hour"]

        # ☀️ Sunlight changes with time
        self.state_data["sun"] = max(0, (12 - abs(hour - 12)) / 12)

        # 🌫 Dirt increases over time
        self.state_data["dirt"] = [min(1, d + 0.05) for d in dirt]

        done = hour >= 24

        return Observation(**self.state_data), Reward(score=reward), done, {}

    def state(self):
        return self.state_data
