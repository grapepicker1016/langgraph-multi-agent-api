from dotenv import load_dotenv
load_dotenv()
import os
import requests 
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool

WEATHER_API = os.getenv('WEATHER_API')



class StatusCodeError(Exception):
    pass

def get_current_weather(query:str):
    """Gets current weather of the given location"""
    try:
        key = os.getenv('WEATHER_API')
    except:
        raise AttributeError("'key' Attribute not defined.")

    # parameters of the api call for current weather
    params = dict(
        q = str(query), # coordinates 
        key = key, # personal key
        aqi = "yes") # air quality data

# https://api.weatherapi.com/v1/current.json?q=london&key=be3dc577705e4b0c9ca72527240603

    base_url = "http://api.weatherapi.com/v1/current.json?" # base url
    url = [str(key)+"="+str(value) for key, value in params.items()] # the rest of the url
    full_url = base_url + "&".join(url) # concatenating

    # making the request
    print(full_url)
    r = requests.get(full_url)

    # Checking response status code
    if r.status_code == 200: # everything its alrighty
        rjson = r.json()
        
        # checking if 'location' and 'current' in rjson

        if "location" not in rjson.keys() or "current" not in rjson.keys():
            raise Exception("location and/or current data not found in the request")
        
        return rjson

    elif r.status_code in [400, 401, 403]:
        error_code = r.json()["error"]["code"]
        error_message = r.json()["error"]["message"]
        raise StatusCodeError(f"{error_message} - Error code: {error_code}")

    else:
        raise Exception("Can't connect to the API")



weather_api = StructuredTool.from_function(
    func=get_current_weather,
    name="weather_api",
    description="Useful when you need to answer questions related to weather, climate and air quality index",
    # coroutine=
    handle_tool_error=True,
)