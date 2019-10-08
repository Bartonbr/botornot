import unittest
from botornot.data import *


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'bidder_id': [1, 2, 2, 3]
        })

    def test_get_total_bids(self):
        result = get_total_bids(self.data)
        self.assertEqual([1,2,1], result['total_bids'].tolist(), "totalbidsdontwork")
