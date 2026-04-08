from fastapi import FastAPI
from env import SolarEnv

app = FastAPI()
env = SolarEnv()

@app.get("/")
def home():
    return {"message": "Solar Env Running"}

@app.api_route("/reset", methods=["GET", "POST"])
def reset(task: str = "easy"):
    return env.reset(task)

@app.get("/state")
def state():
    return env.state()
