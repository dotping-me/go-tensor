package main

import "github.com/dotping-me/go-tensor/autograd/tests"

func main() {

	// ---------------------
	//   Just Tensor stuff
	// ---------------------

	// tests.TestScalarBasics()
	// tests.TestTensorBasics()
	// tests.TestTensorViewOperations()
	// tests.TestMatrixOperations()

	// --------------------
	//    Basic ML stuff
	// --------------------

	// tests.TestReductionOps()
	// tests.TestSoftmax()
	// tests.TestCrossEntropy()
	// tests.TestBatchInference()

	// --------------------
	//    Autograd stuff
	// --------------------

	// tests.TestNumericGrad()
	tests.TestTrainingLoop()
}
