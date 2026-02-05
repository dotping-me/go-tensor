// TODO: Write proper tests

package main

import (
	"fmt"
	"log"

	"github.com/dotping-me/go-tensor/internal/tensor"
)

func main() {
	// T := tensor.NewTensor([]int{3, 1}, []float32{1, 2, 3})

	// // Accessing an element
	// el, err := T.At(1, 0)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// fmt.Printf("Data at index (1, 0): %v\n", el)

	// // Broadcasting -> Resulting shape should be [3 4]
	// broadcastedT, err := tensor.Broadcast(T, []int{1, 4})
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// // Accessing an element in the broadcasted tensor (rank unchanged)
	// el, err = broadcastedT.At(2, 3)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// fmt.Printf("Data at index (2, 3): %v\n", el)

	// // Adding 2 tensors
	// A := tensor.NewTensor([]int{3, 1}, []float32{1, 2, 3})
	// B := tensor.NewTensor([]int{1, 4}, []float32{10, 20, 30, 40})
	// C, err := tensor.Add(A, B)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// fmt.Printf("%v\n", C)

	// // Test reshape
	// D := tensor.NewTensor([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	// E, err := tensor.Reshape(D, []int{3, 2})
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// fmt.Printf("%v\n", E)

	// // Test flatten
	// F := tensor.Flatten(D)
	// fmt.Printf("%v\n", F) // NOTE: Logic for string representation is broken

	// // Test transpose
	// G := tensor.NewTensor([]int{2, 3, 4}, make([]float32, 2*3*4))
	// Gt, err := tensor.Transpose(G, []int{2, 0, 1})
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// fmt.Printf("%v\n", Gt)

	// // Test 2D matrix mult
	// H := tensor.NewTensor([]int{2, 2}, []float32{1, 2, 3, 4})
	// I := tensor.NewTensor([]int{2, 3}, []float32{5, 6, 7, 8, 9, 0})
	// J, err := tensor.Matrix2DMul(H, I)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// fmt.Printf("%v\n", J)

	// // Test Exp()
	// K, err := tensor.Exp(A)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// fmt.Printf("%v\n", K)

	// // Test slicing
	// // B = [ [10 20 30 40] ]
	// BSlice, err := B.SlicePerAxis(tensor.All(), tensor.Slice{
	// 	Start: 1,
	// 	End:   3,
	// 	Step:  1,
	// })
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// fmt.Println(BSlice)

	// Testing batched matrix multiplication
	A := tensor.NewTensor([]int{2, 1, 2, 3}, []float32{

		// Batch 1
		1, 2, 3,
		4, 5, 6,

		// Batch 2
		7, 8, 9,
		10, 11, 12,
	})

	B := tensor.NewTensor([]int{1, 3, 3, 2}, []float32{

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

	C, err := tensor.BatchedMatrixMul(A, B)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(C)
}
