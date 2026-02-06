package tests

import (
	"fmt"

	"github.com/dotping-me/go-tensor/internal/tensor"
)

func TestLinearRegression() {
	fmt.Println("\n\n----- Test Linear Reg #1 -----")
	fmt.Println("           y = XW + b")

	/*
		    | 1 2 |   |  0.5 |
		y = | 3 4 | x | -1.0 |   + | 2 |
		    | 5 6 |
	*/

	X := tensor.NewTensor([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	W := tensor.NewTensor([]int{2, 1}, []float32{0.5, -1.0})
	b := tensor.NewTensor([]int{1}, []float32{2.0})

	XW, err := X.Matrix2DMul(W)
	if err != nil {
		panic(err)
	}

	Y, err := XW.Add(b)
	if err != nil {
		panic(err)
	}

	fmt.Println("\n", Y)
}

func TestSoftmax() {
	fmt.Println("\n\n----- Test Softmax #1 -----")
	fmt.Println("softmax(x_i) = e**x_i / (∑_j (e**x_i))")

	X := tensor.NewTensor([]int{1, 3}, []float32{1, 2, 3})
	S, err := X.Softmax(1)
	if err != nil {
		panic(err)
	}

	fmt.Println("\n", S)
}

func TestCrossEntropy() {
	fmt.Println("\n\n----- Test Cross Entropy #1 -----")
	fmt.Println("L = - ∑ ( y * log( softmax(x) ))")

	logits := tensor.NewTensor([]int{1, 3}, []float32{1, 2, 3})
	labels := tensor.NewTensor([]int{1, 3}, []float32{0, 0, 1})

	loss, err := tensor.CrossEntropy(logits, labels, 1)
	if err != nil {
		panic(err)
	}

	fmt.Println("\n", loss)
}

func TestBatchInference() {
	fmt.Println("\n\n----- Test Batch Inference #1 -----")
	fmt.Println("Batches of linear predictions -> Plotted over a Normal Distribution")

	X := tensor.NewTensor([]int{4, 3}, []float32{
		1, 0, 1,
		0, 1, 1,
		1, 1, 0,
		0, 0, 1,
	})

	W := tensor.NewTensor([]int{3, 2}, []float32{
		0.2, -0.5,
		1.0, 0.3,
		-0.7, 0.8,
	})

	logits, err := X.Matrix2DMul(W)
	if err != nil {
		panic(err)
	}

	probs, err := logits.Softmax(1)
	if err != nil {
		panic(err)
	}

	fmt.Println("\n", probs)
}
