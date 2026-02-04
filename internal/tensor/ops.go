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
	return elementwiseScalarOperation(
		a, b, func(x, y float32) float32 { return x + y },
	)
}

func Sub(a, b *Tensor) (*Tensor, error) {
	return elementwiseScalarOperation(
		a, b, func(x, y float32) float32 { return x - y },
	)
}

func Mul(a, b *Tensor) (*Tensor, error) {
	return elementwiseScalarOperation(
		a, b, func(x, y float32) float32 { return x * y },
	)
}

func Div(a, b *Tensor) (*Tensor, error) {
	return elementwiseScalarOperation(
		a, b, func(x, y float32) float32 { return x / y },
	)
}

func elementwiseScalarOperation(
	a, b *Tensor,
	callback func(float32, float32) float32,
) (*Tensor, error) {

	// Finds the output shape first
	outputShape, err := FindBroadcastShape(a.shape, b.shape)
	if err != nil {
		return nil, err
	}

	// Broadcasts A and B
	tbA, err := Broadcast(a, outputShape)
	if err != nil {
		return nil, err
	}

	tbB, err := Broadcast(b, outputShape)
	if err != nil {
		return nil, err
	}

	// Creates output tensor
	outputTensor := NewTensor(
		outputShape, make([]float32, numberOfDataAsPerShape(outputShape)),
	)

	// Performs operation iteratively
	walkerA := newWalker(tbA)
	walkerB := newWalker(tbB)
	walkerOut := newWalker(outputTensor)

	for {
		outputTensor.data[walkerOut.offset] = callback(
			walkerA.Value(), walkerB.Value())

		// Checks if the entire tensor was traversed
		if !walkerOut.WalkOne() {
			break
		}

		walkerA.WalkOne()
		walkerB.WalkOne()
	}

	return outputTensor, nil
}
