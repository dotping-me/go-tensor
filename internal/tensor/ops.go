package tensor

// Finds the number of data elements according to shape
func numberOfDataAsPerShape(shape []int) int {
	total := 1
	for _, axis := range shape {
		total *= axis
	}

	return total
}

// Adds 2 tensors together
func Add(a, b *Tensor) (*Tensor, error) {
	outputShape, err := FindBroadcastShape(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}

	// Broadcast A and B
	tbA, err := Broadcast(a, outputShape)
	if err != nil {
		return nil, err
	}

	tbB, err := Broadcast(b, outputShape)
	if err != nil {
		return nil, err
	}

	// Create output tensor
	outputTensor := NewTensor(
		outputShape, make([]float32, numberOfDataAsPerShape(outputShape)),
	)

	// Perform addition iteratively
	walkerA := NewWalker(tbA)
	walkerB := NewWalker(tbB)
	walkerOut := NewWalker(outputTensor)

	for {
		outputTensor.Data[walkerOut.offset] = walkerA.Value() + walkerB.Value()

		// Checks if the entire tensor was traversed
		if !walkerOut.WalkOne() {
			break
		}

		walkerA.WalkOne()
		walkerB.WalkOne()
	}

	return outputTensor, nil
}
