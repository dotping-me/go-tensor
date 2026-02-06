package tensor

import (
	"fmt"
)

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
