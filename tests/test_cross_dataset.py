import numpy as np
import pandas as pd

from sentinel.ids.cross_dataset import canonical, normalize_label, shared_feature_xy


def test_canonical_unifies_both_naming_conventions() -> None:
    # 2017 long names and 2018 abbreviations must collapse to one key.
    assert canonical("Total Fwd Packet") == canonical("Tot Fwd Pkts")
    assert canonical("Fwd Packet Length Max") == canonical("Fwd Pkt Len Max")
    assert canonical("Bwd Segment Size Avg") == canonical("Bwd Seg Size Avg")


def test_label_aliasing_maps_2018_to_2017() -> None:
    assert normalize_label("FTP-BruteForce") == "FTP-PATATOR"
    assert normalize_label("SSH-Bruteforce") == "SSH-PATATOR"
    assert normalize_label("Benign") == "BENIGN"


def test_shared_feature_xy_intersects_and_drops_ids() -> None:
    train = pd.DataFrame(
        {
            "Total Fwd Packet": [1.0, 2.0, 3.0, 4.0],
            "Flow Duration": [10.0, 20.0, 30.0, 40.0],
            "Dst IP": ["a", "b", "c", "d"],  # identifier — must be dropped
            "Label": ["BENIGN", "FTP-Patator", "BENIGN", "FTP-Patator"],
        }
    )
    test = pd.DataFrame(
        {
            "Tot Fwd Pkts": [1.0, 5.0],
            "Flow Duration": [11.0, 99.0],
            "Extra2018Only": [7.0, 8.0],  # not shared — must be dropped
            "Label": ["Benign", "FTP-BruteForce"],
        }
    )
    from sentinel.ids.cross_dataset import canonical_columns

    x_tr, x_te, y_tr, y_te, labels_te = shared_feature_xy(
        canonical_columns(train), canonical_columns(test)
    )

    assert set(x_tr.columns) == {"totfwdpkt", "flowduration"}
    assert list(x_te.columns) == list(x_tr.columns)
    assert y_tr.tolist() == [0, 1, 0, 1]
    assert y_te.tolist() == [0, 1]
    assert labels_te.tolist() == ["BENIGN", "FTP-PATATOR"]  # aliased
    assert not np.isinf(x_te.to_numpy()).any()
