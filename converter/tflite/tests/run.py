import os
import sys
import unittest

from conv2d.test import Conv2dTestCase

def test_all():
	suite = unittest.TestSuite()
	suite.addTest(Conv2dTestCase('test_single_conv2d'))
	return suite

if __name__ == '__main__':
	print(f"os.path.realpath(__file__) : {os.path.realpath(__file__) }")

	runner = unittest.TextTestRunner()
	runner.run(test_all())