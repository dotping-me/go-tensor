package tensor

import (
	"fmt"
	"math"
)

// Finds the number of data elements according to shape
func numberOfDataAsPerShape(shape []int) int {
	total := 1
	for _, axis := range shape {
		total *= axis
	}

	return total
}

// ---------------------------------------
//   Elementwise operations w/ 1 Tensors!
// ---------------------------------------

func elwiseOpsWithSingleTensor(
	t *Tensor,
	callback func(float32) float32, // TODO: Maybe return an error for sanity?
) (*Tensor, error) {
	numOfUniqueData := len(t.data)

	// Just needs to iterate over data, right? Because shape remains unchanged
	newData := make([]float32, numOfUniqueData)
	for i := range numOfUniqueData {
		newData[i] = callback(t.data[i])
	}

	return NewTensor(t.shape, newData), nil
}

func Exp(t *Tensor) (*Tensor, error) {
	return elwiseOpsWithSingleTensor(
		t, func(x float32) float32 {
			return float32(math.Exp(float64(x)))
		},
	)
}

// ---------------------------------------
//   Elementwise operations w/ 2 Tensors!
// ---------------------------------------

func elwiseOpsWith2Tensors(
	a, b *Tensor,
	callback func(float32, float32) float32,
) (*Tensor, error) {

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

func Add(a, b *Tensor) (*Tensor, error) {
	return elwiseOpsWith2Tensors(
		a, b, func(x, y float32) float32 { return x + y },
	)
}

func Sub(a, b *Tensor) (*Tensor, error) {
	return elwiseOpsWith2Tensors(
		a, b, func(x, y float32) float32 { return x - y },
	)
}

func Mul(a, b *Tensor) (*Tensor, error) {
	return elwiseOpsWith2Tensors(
		a, b, func(x, y float32) float32 { return x * y },
	)
}

func Div(a, b *Tensor) (*Tensor, error) {
	return elwiseOpsWith2Tensors(
		a, b, func(x, y float32) float32 { return x / y },
	)
}

// ------------------------------------
//   Non-elementwise operations below!
//        Proceed with caution!
// ------------------------------------

func Matrix2DMul(a, b *Tensor) (*Tensor, error) {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("Shapes are not 2D: %v and %v", a.shape, b.shape)
	}

	// Checks for if matrix multiplication is valid
	x1, y1 := a.shape[0], a.shape[1]
	x2, y2 := b.shape[0], b.shape[1]

	if y1 != x2 {
		return nil, fmt.Errorf(
			"2D Matrix Multiplication is invalid: y1 (%d) != x2 (%d)", a.shape[1], b.shape[0],
		)
	}

	// Performs multiplication following row by column basis
	newShape := []int{x1, y2}
	newData := []float32{}

	for x := range x1 {
		for y := range y2 {
			sum := float32(0)

			// Iterate through every column of this row in Matrix A
			for z := range y1 {
				elA, err := a.At(x, z)
				if err != nil {
					return nil, err
				}

				elB, err := b.At(z, y)
				if err != nil {
					return nil, err
				}

				sum += elA * elB
			}

			newData = append(newData, sum)
		}
	}

	return NewTensor(newShape, newData), nil
}
