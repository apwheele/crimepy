import pandas as pd
from datetime import datetime, timedelta
from crimepy.chain import NearChains


def test_nearchains_get_clusters():
    """
    Test NearChains.get_clusters with 5 events:
    - 3 events on day 1 (within 12 hours)
    - 2 events on day 3 (within 12 hours)

    Expected: 2 clusters - one with 3 events, one with 2 events
    """
    # Create base datetime
    base_date = datetime(2024, 1, 1, 0, 0, 0)

    # Create dummy data: X=0 for all, Y=range(5)
    # First 3 events on day 1 (within 12 hours)
    # Last 2 events on day 3 (within 12 hours)
    data = pd.DataFrame({
        'x': [0, 0, 0, 0, 0],
        'y': [0, 1, 2, 3, 4],
        'datetime': [
            base_date,                              # Day 1, hour 0
            base_date + timedelta(hours=4),         # Day 1, hour 4
            base_date + timedelta(hours=8),         # Day 1, hour 8
            base_date + timedelta(days=2),          # Day 3, hour 0
            base_date + timedelta(days=2, hours=6)  # Day 3, hour 6
        ]
    })

    # Initialize NearChains
    nc = NearChains(data, x='x', y='y', d='datetime')

    # Get clusters with:
    # - time_thresh=1 day (events must be within 1 day)
    # - space_thresh=1.5 (events must be within 1.5 units spatially)
    clusters = nc.get_clusters(time_thresh=1, space_thresh=1.5)

    # Should have exactly 2 clusters
    assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"

    # First cluster should have 3 events (largest first due to sorting)
    assert len(clusters[0]) == 3, f"Expected first cluster to have 3 events, got {len(clusters[0])}"

    # Second cluster should have 2 events
    assert len(clusters[1]) == 2, f"Expected second cluster to have 2 events, got {len(clusters[1])}"

    # Verify the Y values in first cluster (day 1 events)
    cluster1_y = sorted(clusters[0]['y'].tolist())
    assert cluster1_y == [0, 1, 2], f"Expected first cluster Y values [0, 1, 2], got {cluster1_y}"

    # Verify the Y values in second cluster (day 3 events)
    cluster2_y = sorted(clusters[1]['y'].tolist())
    assert cluster2_y == [3, 4], f"Expected second cluster Y values [3, 4], got {cluster2_y}"
