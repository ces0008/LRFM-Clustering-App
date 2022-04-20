import unittest
from app import lrfm, rank
import yaml


class DateConvert(unittest.TestCase):
    def test_date(self):
        self.assertEqual(lrfm.convert_date(140001, 140005),
                         {140001: 1, 140002: 2, 140003: 3, 140004: 4, 140005: 5})
        self.assertEqual(lrfm.convert_date(140001, 140012),
                         {140001: 1, 140002: 2, 140003: 3, 140004: 4, 140005: 5, 140006: 6,
                          140007: 7, 140008: 8, 140009: 9, 140010: 10, 140011: 11, 140012: 12})
        self.assertEqual(lrfm.convert_date(139909, 140003),
                         {139909: 1, 139910: 2, 139911: 3, 139912: 4, 140001: 5, 140002: 6,
                          140003: 7})


if __name__ == '__main__':
    unittest.main()
