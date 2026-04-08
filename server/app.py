from fastapi import FastAPI
import uvicorn
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


# 🔥 REQUIRED MAIN FUNCTION
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


# 🔥 REQUIRED ENTRY POINT
if __name__ == "__main__":
    main()
