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

// TODO: Maybe turn it into a Tensor method
func Exp(t *Tensor) (*Tensor, error) {
	return elwiseOpsWithSingleTensor(
		t, func(x float32) float32 {
			return float32(math.Exp(float64(x)))
		},
	)
}

func Log(t *Tensor) (*Tensor, error) {
	return elwiseOpsWithSingleTensor(
		t, func(x float32) float32 {
			return float32(math.Log(float64(x)))
		},
	)
}

func Neg(t *Tensor) (*Tensor, error) {
	return elwiseOpsWithSingleTensor(
		t, func(x float32) float32 {
			return -x
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

func BatchedMatrixMul(a, b *Tensor) (*Tensor, error) {
	if len(a.shape) < 2 || len(b.shape) < 2 {
		return nil, fmt.Errorf("Both Tensors ranks must be >= 2!")
	}

	// Seperate 2D matrices from tensors
	batchA := a.shape[:len(a.shape)-2]
	batchB := b.shape[:len(b.shape)-2]

	matrixAx, matrixAy := a.shape[len(a.shape)-2], a.shape[len(a.shape)-1]
	matrixBx, matrixBy := b.shape[len(b.shape)-2], b.shape[len(b.shape)-1]
	if matrixAy != matrixBx {
		return nil, fmt.Errorf(
			"2D Matrix Multiplication is invalid: y1 (%d) != x2 (%d)", matrixAy, matrixBx,
		)
	}

	// Finds the broadcast shape if possible
	broadcastedBatchShape, err := findBroadcastShape(batchA, batchB)
	if err != nil {
		return nil, err
	}

	// Broadcast each original tensor to the common broadcast shape + their 2D matrices
	// This is done to make the number of batches in both tensors the same so
	// that multiplication between batches to batches is possible

	bA, err := Broadcast(a, append(broadcastedBatchShape, matrixAx, matrixAy))
	if err != nil {
		return nil, err
	}

	bB, err := Broadcast(b, append(broadcastedBatchShape, matrixBx, matrixBy))
	if err != nil {
		return nil, err
	}

	// Prepare outputs
	outputShape := append(broadcastedBatchShape, matrixAx, matrixBy)
	outputData := []float32{}

	batchWalker := newShapeWalker(broadcastedBatchShape)
	for !batchWalker.done {

		// Extract batches
		// A batch will be basically the size of (Matrix Ax x Matrix By)
		batchIndex := batchWalker.Index()

		batchA, err := bA.Get2DSliceAtParentIndex(batchIndex)
		if err != nil {
			return nil, err
		}

		batchB, err := bB.Get2DSliceAtParentIndex(batchIndex)
		if err != nil {
			return nil, err
		}

		// Perform multiplication
		batchC, err := Matrix2DMul(batchA, batchB)
		if err != nil {
			return nil, err
		}

		outputData = append(outputData, batchC.data...)
		batchWalker.WalkOne()
	}

	return NewTensor(outputShape, outputData), nil
}
