import os
import sys
import unittest

from conv2d.test import Conv2dTestCase
from fcnn_relu.test import FcnnReluTestCase
from fcnn_sigmoid.test import FcnnSigmTestCase


def test_all():
	suite = unittest.TestSuite()
	suite.addTest(Conv2dTestCase('test_single_conv2d'))
	suite.addTest(FcnnReluTestCase('test_codegen'))
	suite.addTest(FcnnSigmTestCase('test_codegen'))
	return suite

if __name__ == '__main__':
	print(f"os.path.realpath(__file__) : {os.path.realpath(__file__) }")

	runner = unittest.TextTestRunner()
	runner.run(test_all())