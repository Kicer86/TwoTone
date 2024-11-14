
import unittest

import utils


class TestUTUtils(unittest.TestCase):

    def test_around_in_epsilon(self):
        f5_0 = utils.Around(5.0, 0.5)

        self.assertEqual(f5_0, 5.0)
        self.assertEqual(f5_0, 5.4)
        self.assertEqual(f5_0, 4.6)

    def test_around_not_in_epsilon(self):
        f5_0 = utils.Around(5.0, 0.5)

        self.assertNotEqual(f5_0, 5.6)
        self.assertNotEqual(f5_0, 4.4)


if __name__ == '__main__':
    unittest.main()
