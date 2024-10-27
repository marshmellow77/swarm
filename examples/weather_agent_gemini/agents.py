import json
import os
import requests
from dotenv import load_dotenv

from swarm import Agent

# Load environment variables
load_dotenv()

def get_weather(location, time="now"):
    """Get the current weather in a given location. Location MUST be a city."""
    api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    
    if api_key:
        try:
            response = requests.get(
                "http://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": location,
                    "appid": api_key,
                    "units": "metric"
                }
            )
            data = response.json()
            return json.dumps({
                "location": data["name"],
                "temperature": data["main"]["temp"],
                "time": time
            })
        except Exception as e:
            print(f"API call failed: {e}. Using mock data.")
    
    # Mock response as fallback
    return json.dumps({
        "location": location,
        "temperature": "65",
        "time": time
    })

weather_agent = Agent(
    name="Weather Agent",
    instructions="You are a helpful agent.",
    functions=[get_weather],
)
