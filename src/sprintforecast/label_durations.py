from datetime import datetime
import numpy as np
from .timeline_fetcher import TimelineFetcher
from .github_client import GitHubClient

def extract_label_durations(
    gh: GitHubClient,
    owner: str,
    repo: str,
    nums: list[int],
):
    tf = TimelineFetcher(gh, owner, repo)
    dev, rev = [], []
    for n in nums:
        di = do = ri = ro = None
        for ev in tf.iter_events(n, types=['labeled', 'unlabeled']):
            lab = ev.get('label', {}).get('name', '').lower()
            ts  = ev['created_at']
            if lab == 'dev':
                if ev['event'] == 'labeled':   di = ts
                else:                          do = ts
            if lab == 'review':
                if ev['event'] == 'labeled':   ri = ts
                else:                          ro = ts
        if di and do:
            dev.append(
                (datetime.fromisoformat(do.rstrip('Z')) -
                 datetime.fromisoformat(di.rstrip('Z'))).total_seconds()/3600
            )
        if ri and ro:
            rev.append(
                (datetime.fromisoformat(ro.rstrip('Z')) -
                 datetime.fromisoformat(ri.rstrip('Z'))).total_seconds()/3600
            )
    return np.asarray(dev), np.asarray(rev)
