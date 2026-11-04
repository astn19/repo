from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
LINES = ROOT / "data" / "lines"
REPORTS = ROOT / "reports"

RELEASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
TEAM_BOX_URL = RELEASE + "/espn_mens_college_basketball_team_boxscores/team_box_{season}.parquet"
SCHEDULE_URL = RELEASE + "/espn_mens_college_basketball_schedules/mbb_schedule_{season}.parquet"

# Season = year the season ends (2024 = 2023-24).
FIRST_SEASON = 2010
LAST_SEASON = 2027  # in-progress season

# Teams with fewer than this many games in a season are treated as non-D1
# and pooled into one "OTHER" opponent.
MIN_GAMES_D1 = 10

# Standard -110 pricing.
JUICE_ODDS = -110
BREAKEVEN = 110 / 210  # 0.5238
