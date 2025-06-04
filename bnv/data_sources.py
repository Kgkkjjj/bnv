from __future__ import annotations

import requests

USER_AGENT = "bnv-data-fetcher/1.0"


def fetch_wikipedia_summary(topic: str) -> str:
    """Fetch a short summary from Wikipedia for the given topic."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("extract", "")


def fetch_random_quote() -> str:
    """Return a random quote from the Quotable API."""
    resp = requests.get("https://api.quotable.io/random", headers={"User-Agent": USER_AGENT}, timeout=10)
    resp.raise_for_status()
    return resp.json().get("content", "")


def fetch_random_activity() -> str:
    """Get a random activity suggestion from the Bored API."""
    resp = requests.get("https://www.boredapi.com/api/activity", headers={"User-Agent": USER_AGENT}, timeout=10)
    resp.raise_for_status()
    return resp.json().get("activity", "")


def fetch_random_user() -> str:
    """Fetch a short description of a random user from randomuser.me."""
    resp = requests.get("https://randomuser.me/api/", headers={"User-Agent": USER_AGENT}, timeout=10)
    resp.raise_for_status()
    data = resp.json()["results"][0]
    first = data["name"]["first"]
    last = data["name"]["last"]
    city = data["location"]["city"]
    return f"{first} {last} from {city}"


def fetch_weather(latitude: float, longitude: float) -> str:
    """Retrieve current weather information from the Open-Meteo API."""
    url = (
        "https://api.open-meteo.com/v1/forecast?latitude="
        f"{latitude}&longitude={longitude}&current=temperature_2m"
    )
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    temp = data.get("current", {}).get("temperature_2m")
    if temp is None:
        return ""
    return f"Current temperature: {temp}°C"


def fetch_live_data(topics: list[str] | None = None) -> list[str]:
    """Collect text samples from several open APIs."""
    topics = topics or ["Artificial_intelligence"]
    samples = []
    for topic in topics:
        samples.append(fetch_wikipedia_summary(topic))
    samples.append(fetch_random_quote())
    samples.append(fetch_random_activity())
    samples.append(fetch_random_user())
    samples.append(fetch_weather(35.0, 139.0))  # default coordinates (Tokyo)
    return [s for s in samples if s]
