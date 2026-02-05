// TODO: Write proper tests

package main

import (
	"github.com/dotping-me/go-tensor/internal/tensor/tests"
)

func main() {
	tests.TestScalarBasics()
	tests.TestTensorBasics()
	tests.TestTensorViewOperations()
	tests.TestMatrixOperations()
}
