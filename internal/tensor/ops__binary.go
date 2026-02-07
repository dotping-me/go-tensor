// Arithmetic operations involving 1+ tensors!

package tensor

func elwiseOpsWith2Tensors(
	a, b *Tensor,
	callback func(float32, float32) float32,
) (*Tensor, error) {

	// Checks if one of the tensors is a scalar
	if len(a.shape) == 0 {
		return elwiseOpsWithSingleTensor(
			b, func(x float32) float32 {
				return callback(a.data[0], x)
			},
		)
	}

	if len(b.shape) == 0 {
		return elwiseOpsWithSingleTensor(
			a, func(x float32) float32 {
				return callback(b.data[0], x)
			},
		)
	}

	// Finds the output shape first
	outputShape, err := findBroadcastShape(a.shape, b.shape)
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
		outputShape, make([]float32, NumberOfDataAsPerShape(outputShape)),
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

func (a *Tensor) Add(b *Tensor) (*Tensor, error) {
	return elwiseOpsWith2Tensors(
		a, b, func(x, y float32) float32 { return x + y },
	)
}

func (a *Tensor) Sub(b *Tensor) (*Tensor, error) {
	return elwiseOpsWith2Tensors(
		a, b, func(x, y float32) float32 { return x - y },
	)
}

func (a *Tensor) Mul(b *Tensor) (*Tensor, error) {
	return elwiseOpsWith2Tensors(
		a, b, func(x, y float32) float32 { return x * y },
	)
}

func (a *Tensor) Div(b *Tensor) (*Tensor, error) {
	return elwiseOpsWith2Tensors(
		a, b, func(x, y float32) float32 { return x / y },
	)
}
