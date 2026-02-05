package tests

import (
	"fmt"
	"log"

	"github.com/dotping-me/go-tensor/internal/tensor"
)

func TestTensorBasics() {
	fmt.Println("\n\n----- Test Tensor Basic #1 -----")
	fmt.Println("Creating tensor...")

	T := tensor.NewTensor([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	fmt.Println("\nCreating a Tensor T:\n", T)

	fmt.Println("\n\n----- Test Tensor Basic #2 -----")
	fmt.Println("Data access...")

	// Accessing an element
	el, err := T.At(1, 1)
	if err != nil {
		panic(err)
	}

	fmt.Println("Element of T at (1, 1):", el) // Should be 5
}

func TestTensorViewOperations() {
	fmt.Println("\n\n----- Test Tensor View #1 -----")
	fmt.Println("Brocasting tensor...")

	// Broadcast T to shape [1, 3]
	T := tensor.NewTensor([]int{3, 1}, []float32{1, 2, 3})
	bT, err := tensor.Broadcast(T, []int{1, 4}) // Shape should be [3 4]
	if err != nil {
		panic(err)
	}

	fmt.Println("\nBroadcasted [[1 2 3]] to shape [3 1]:\n", bT)

	fmt.Println("\n\n----- Test Tensor View #2 -----")
	fmt.Println("Data access in broadcasted tensor...")

	// Accessing an element in the broadcasted tensor (rank unchanged)
	el, err := bT.At(2, 3)
	if err != nil {
		panic(err)
	}

	fmt.Println("Element of bT at (2, 3):", el) // Should be 3

	fmt.Println("\n\n----- Test Tensor View #3 -----")
	fmt.Println("Reshaping a tensor...")

	T = tensor.NewTensor([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	rsT, err := tensor.Reshape(T, []int{3, 2})
	if err != nil {
		panic(err)
	}

	fmt.Printf("\nReshaping %v to\n%v\n", T, rsT)

	fmt.Println("\n\n----- Test Tensor View #4 -----")
	fmt.Println("Flattening a tensor...")

	fT := tensor.Flatten(T)
	fmt.Printf("\n[[1 2 3] [4 5 6]] flattened to:\n%v\n", fT) // NOTE: Logic for string representation is broken

	fmt.Println("\n\n----- Test Tensor View #5 -----")
	fmt.Println("Transposing a tensor...")

	tT, err := tensor.Transpose(T, []int{1, 0})
	if err != nil {
		panic(err)
	}

	fmt.Printf("\nTransposed [[1 2 3] [4 5 6]] over Axis Indices [1 0]:\n%v\n", tT)
}

func TestTensorUnaryOperations() {
	fmt.Println("\n\n----- Test Tensor Unary Op #1 -----")
	fmt.Println("Doing exponential of a tensor...")

	T := tensor.NewTensor([]int{2, 2}, []float32{1, 2, 3, 4})
	eT, err := tensor.Exp(T)
	if err != nil {
		panic(err)
	}

	fmt.Printf("\nDoing exponential of [[1 2] [3 4]]:\n%v\n", eT)
}

func TestTensorBinaryOperations() {
	fmt.Println("\n\n----- Test Tensor Binary Op #1 -----")
	fmt.Println("Adding 2 tensors...")

	// Adding 2 tensors
	A := tensor.NewTensor([]int{3, 1}, []float32{1, 2, 3})
	B := tensor.NewTensor([]int{1, 4}, []float32{10, 20, 30, 40})
	C, err := tensor.Add(A, B)
	if err != nil {
		panic(err)
	}

	fmt.Printf("\n%v\n+\n%v\n=\n%v\n", A, B, C)
}

func TestMatrixOperations() {
	fmt.Println("\n\n----- Test Tensor Matrix Op #1 -----")
	fmt.Println("Multiplying 2 2D matrices...")

	A := tensor.NewTensor([]int{2, 2}, []float32{1, 2, 3, 4})
	B := tensor.NewTensor([]int{2, 3}, []float32{5, 6, 7, 8, 9, 0})
	C, err := tensor.Matrix2DMul(A, B)
	if err != nil {
		panic(err)
	}

	fmt.Printf("\n%v\nx\n%v\n=\n%v\n", A, B, C)

	fmt.Println("\n\n----- Test Tensor Matrix Op #2 (Sort of but not really) -----")
	fmt.Println("Slicing...")

	// Test slicing
	// B = [ [5 6 7] [8 9 10] ]
	sB, err := B.SlicePerAxis(tensor.All(), tensor.Slice{
		Start: 1,
		End:   3,
		Step:  1,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\nSlicing [[5 6 7] [8 9 10]] along [:, 1:3, 1]:\n%v\n", sB)

	fmt.Println("\n\n----- Test Tensor Matrix Op #3 -----")
	fmt.Println("Batched matrix multiplication...")

	A = tensor.NewTensor([]int{2, 1, 2, 3}, []float32{

		// Batch 1
		1, 2, 3,
		4, 5, 6,

		// Batch 2
		7, 8, 9,
		10, 11, 12,
	})

	B = tensor.NewTensor([]int{1, 3, 3, 2}, []float32{

		// Batch 1
		1, 2,
		3, 4,
		5, 6,

		// Batch 2
		7, 8,
		9, 10,
		11, 12,

		// Batch 3
		13, 14,
		15, 16,
		17, 18,
	})

	C, err = tensor.BatchedMatrixMul(A, B)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(`
	[ [[1 2 3] [4 5 6]] 
	  [[7 8 9] [10 11 12]] ]

	             X

	[ [[1   2] [3   4] [5   6]]
	  [[7   8] [9  10] [11 12]]
	  [[13 14] [15 16] [17 18]] ]
	
	             =`)
	fmt.Println(C)
}

func TestScalarBasics() {
	fmt.Println("\n\n----- Test Scalar #1 -----")
	fmt.Println("Adding 2 Scalars...")

	// Adding 2 scalars
	S1 := tensor.NewTensor([]int{}, []float32{5})
	S2 := tensor.NewTensor([]int{}, []float32{1})
	Sum, _ := tensor.Add(S1, S2)
	fmt.Println("\nAdding scalar 5 with scalar 1:", Sum)

	fmt.Println("\n\n----- Test Scalar #2 -----")
	fmt.Println("Multiplying 2 Scalars...")

	// Multiplying a scalar with a tensor
	T1 := tensor.NewTensor([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	Mul, _ := tensor.Mul(T1, S1)
	fmt.Println("\nMultiplying scalar 5 with tensor [ [1 2 3] [4 5 6] ]:\n", Mul)

	Exp, _ := tensor.Exp(S1)
	fmt.Println(Exp)
}

func TestReductionOps() {
	fmt.Println("\n\n----- Test Reduction Ops #1 -----")
	fmt.Println("Summation along axis, consuming it...")

	T := tensor.NewTensor([]int{3, 3}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	rsT, err := T.Sum(1, false)
	if err != nil {
		panic(err)
	}

	fmt.Printf("[[1 2 3] [4 5 6] [7 8 9]] summed along axis 1:\n%v\n", rsT)
}
