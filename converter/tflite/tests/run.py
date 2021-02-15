import os
import sys
import unittest

from conv2d.test import Conv2dTestCase
from maxpool2d.test import MaxPool2DTestCase
from fcnn_relu.test import FcnnReluTestCase
from fcnn_tanh.test import FcnnTanhTestCase
from fcnn_sigmoid.test import FcnnSigmTestCase
from fcnn_double_relu.test import FcnnDoubleReluTestCase


def test_all():
	suite = unittest.TestSuite()
	suite.addTest(Conv2dTestCase('test_single_conv2d'))
	suite.addTest(MaxPool2DTestCase('test_codegen'))
	suite.addTest(FcnnReluTestCase('test_codegen'))
	suite.addTest(FcnnSigmTestCase('test_codegen'))
	suite.addTest(FcnnTanhTestCase('test_codegen'))
	suite.addTest(FcnnDoubleReluTestCase('test_codegen'))
	return suite

if __name__ == '__main__':
	runner = unittest.TextTestRunner()
	runner.run(test_all())