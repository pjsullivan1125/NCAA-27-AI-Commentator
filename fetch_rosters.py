"""Download both rosters from the team sites into rosters/*.json. Run once, and again if rosters change."""
import json
import re
import sys
import requests
from bs4 import BeautifulSoup
import config

URLS = {
    "msu": "https://msuspartans.com/sports/football/roster",
    "akron": "https://gozips.com/sports/football/roster",
}
TEXT_RE = re.compile(r"Jersey Number\s*(\d+)\s*([A-Za-z][A-Za-z.'\u2019,\- ]*?)\s*Position\s*([A-Z/]+)")


def parse_text(html):
    text = BeautifulSoup(html, "html.parser").get_text(" ")
    seen, players = set(), []
    for num, name, pos in TEXT_RE.findall(text):
        name = re.sub(r"\s+", " ", name).strip()
        key = (int(num), name)
        if key not in seen:
            seen.add(key)
            players.append({"number": int(num), "name": name, "pos": pos})
    return players


def parse_classes(html):
    soup = BeautifulSoup(html, "html.parser")
    seen, players = set(), []
    for li in soup.select("li[class*=sidearm-roster-player]"):
        def grab(fragment):
            el = li.select_one(f"[class*={fragment}]")
            return el.get_text(" ", strip=True) if el else ""
        num = re.search(r"\d+", grab("jersey-number"))
        name, pos = grab("player-name"), grab("player-position").split(" ")[0]
        if num and name and pos:
            key = (int(num.group()), name)
            if key not in seen:
                seen.add(key)
                players.append({"number": int(num.group()), "name": name, "pos": pos})
    return players


def main():
    config.ROSTER_DIR.mkdir(exist_ok=True)
    for team, url in URLS.items():
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        players = parse_text(html) or parse_classes(html)
        if len(players) < 60:
            print(f"[!] {team}: only found {len(players)} players. The site layout may have changed.")
        players.sort(key=lambda p: (p["number"], p["name"]))
        (config.ROSTER_DIR / f"{team}.json").write_text(json.dumps(players, indent=1), encoding="utf-8")
        print(f"{team}: saved {len(players)} players")


if __name__ == "__main__":
    sys.exit(main())
