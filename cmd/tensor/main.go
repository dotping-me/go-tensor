package main

import (
	"fmt"
	"log"

	"github.com/dotping-me/go-tensor/internal/tensor"
)

func main() {
	T := tensor.NewTensor([]int{3, 1}, []float32{1, 2, 3})

	// Accessing an element
	el, err := T.At(1, 0)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Data at index (1, 0): %v\n", el)

	// Broadcasting -> Resulting shape should be [3 4]
	broadcastedT, err := tensor.Broadcast(T, []int{1, 4})
	if err != nil {
		log.Fatal(err)
	}

	// Accessing an element in the broadcasted tensor (rank unchanged)
	el, err = broadcastedT.At(2, 3)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Data at index (2, 3): %v\n", el)

	test, err := tensor.FindBroadcastShape([]int{3, 1}, []int{1, 4, 7})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%v\n", test)

	// Adding 2 tensors
	A := tensor.NewTensor([]int{3, 1}, []float32{1, 2, 3})
	B := tensor.NewTensor([]int{1, 4}, []float32{10, 20, 30, 40})
	C, err := tensor.Add(A, B)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%v\n", C.Data)
}
