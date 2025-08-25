
import json
from typing import Dict, Any, Optional
import requests
import os
from dotenv import load_dotenv


load_dotenv()
#DANGEROUSLY_OMIT_AUTH= "true"



WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "261e8bc58bc0939688b5be0029e64fa8")


def perform_websearch(query: str) -> str:
    """
    Performs a web search using SerpAPI
    Args:
        query: the search query
    """
    try:
        api_key = "7ae12f51127051b3f3d75975e8e1bb29bdea9ef1921d3e568411c907d4f69956"
        
        if not api_key:
            return json.dumps({"error": "Missing SERPAPI_KEY environment variable"})
        
        url = "https://serpapi.com/search"
        params = {
            'engine': 'google',
            'q': query,
            'api_key': api_key,
            'num': 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if 'error' in data:
            return json.dumps({"error": f"SerpAPI error: {data['error']}"})
        
        if 'organic_results' not in data or not data['organic_results']:
            return json.dumps({"error": f"No search results found for: {query}"})
        
        results = []
        for item in data['organic_results']:
            results.append({
                "title": item.get('title', 'N/A'),
                "url": item.get('link', 'N/A'),
                "snippet": item.get('snippet', 'N/A')
            })
        
        return json.dumps({"results": results, "query": query})
        
    except requests.exceptions.Timeout:
        return json.dumps({"error": "Search request timed out"})
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Network error: {str(e)}"})
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON response from SerpAPI"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})



def get_weather(city: str) -> Dict[str, Any]:
    """
    Get current weather for a city.
    Returns predefined mock data for major cities due to firewall restrictions.
    
    Args:
        city: City name (supports New York, Mumbai, Delhi, Chennai)
    
    Returns:
        Dictionary containing weather information in Celsius
    """
    try:
        if not city.strip():
            return {"error": "City name cannot be empty", "success": False}
        
        
        weather_data = {
            "newyork": {
                "city": "New York",
                "country": "US",
                "temperature": 22,
                "feels_like": 25,
                "humidity": 60,
                "pressure": 1015,
                "description": "clear sky",
                "wind_speed": 15
            },
            "mumbai": {
                "city": "Mumbai", 
                "country": "IN",
                "temperature": 31,
                "feels_like": 36,
                "humidity": 78,
                "pressure": 1009,
                "description": "humid and warm",
                "wind_speed": 8
            },
            "delhi": {
                "city": "Delhi",
                "country": "IN", 
                "temperature": 28,
                "feels_like": 32,
                "humidity": 65,
                "pressure": 1012,
                "description": "partly cloudy",
                "wind_speed": 12
            },
            "chennai": {
                "city": "Chennai",
                "country": "IN",
                "temperature": 33,
                "feels_like": 38,
                "humidity": 72,
                "pressure": 1008,
                "description": "hot and sunny",
                "wind_speed": 10
            }
        }
        
     
        city_key = city.lower().strip().replace(" ", "").replace("new york", "newyork")
        
        if city_key in weather_data:
            data = weather_data[city_key]
            
            weather_info = {
                "success": True,
                "city": data["city"],
                "country": data["country"],
                "temperature": data["temperature"],
                "feels_like": data["feels_like"],
                "humidity": data["humidity"],
                "pressure": data["pressure"],
                "description": data["description"],
                "wind_speed": data["wind_speed"],
                "units": "Celsius",
                "note": "Mock data - firewall environment"
            }
            
            return weather_info
        else:
            return {
                "error": f"City '{city}' not supported. Available cities: New York, Mumbai, Delhi, Chennai",
                "success": False,
                "available_cities": ["New York", "Mumbai", "Delhi", "Chennai"]
            }
        
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "success": False}