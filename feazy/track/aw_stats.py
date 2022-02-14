"""
Get stats from activity watch
"""
from datetime import datetime, time, timedelta
import pdb
from aw_client import ActivityWatchClient
import socket

# AFK bucet
# bucket_id = f"aw-watcher-afk_{socket.gethostname()}"
afk_bucket_id = "aw-watcher-afk_Kobis-MacBook-Pro.local"
window_bucket_id = "aw-watcher-window_Kobis-MacBook-Pro.local"


def work_time_series(awc: ActivityWatchClient, start_time, end_time):
    events = awc.get_events(bucket_id=bucket_id, start=start_time, end=end_time)
    import pdb

    pdb.set_trace()
    events = [
        e
        for e in events
        if e.data["status"] == "not-afk" and e.data["category"] == "Work"
    ]
    total_duration = sum((e.duration for e in events), timedelta())
    return total_duration


if __name__ == "__main__":
    awc = ActivityWatchClient("testclient")
    start_time = datetime(2022, 2, 7, 0, 0)
    end_time = start_time + timedelta(days=4)
    total_duration = work_time_series(awc, start_time, end_time)
    print(
        f"Total time spent on computer today: {total_duration.total_seconds()/3600:.02f} hours."
    )
