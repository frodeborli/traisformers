import requests

def resolve_radio(radio: str) -> int:
    """
    Resolve the MMSI of a vessel based on its radio callsign, and return the MMSI
    """
    url = f"https://kystdatahuset.no/ws/api/ship/combined/callsign/{radio}"

    r = requests.get(url)
    if r.status_code != 200:
        print(f"[!] Could not resolve mmsi from radio ({radio})")
        print(r.status_code)
        print(r.text)
        return None
    try:
        res = r.json()
    except:
        print("[!] Could not parse JSON response body")
        print(r.text)
        return None

    if not res.get("success") == True or res.get("data") == None:
        print(f"[!] Error in response to resolve mmsi from radio ({radio})")
        print(r.text)
        return None

    # If multiple entries returned, get the non-zero MMSI
    for msg in res.get("data"):
        if msg.get("mmsi", 0) <= 0:
            continue
        return msg.get("mmsi")