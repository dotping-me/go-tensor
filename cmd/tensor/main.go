package main

import (
	"fmt"
	"log"

	"github.com/dotping-me/go-tensor/internal/tensor"
)

func main() {
	T := tensor.New([]int{3, 1}, []float32{1, 2, 3})

	// Accessing an element
	el, err := T.At(1, 0)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Data at index (1, 0): %v\n", el)
}
