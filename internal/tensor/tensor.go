package tensor

import "fmt"

type Tensor struct {
	Shape   []int
	Data    []float32
	Strides []int
}

// Calculates the strides for a tensor
// A stride is basically the # of elements that needs to be moved per axis to
// access the next batch
func CalculateStrides(shape []int) []int {
	strides := make([]int, len(shape))

	// Formula:
	// s (of N) = d (of N + 1) * s (of N + 1)

	// NOTE: The stride for the last axis will always be 1

	// It makes much more sense to iteracte over the shape backwards
	var s int = 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = s
		s *= shape[i] // d (of N+1) * s (of N+1) = stride for the previous dimension
	}

	return strides
}

// Constructor which calculates strides on instantiation
func New(shape []int, data []float32) *Tensor {
	return &Tensor{
		Shape:   shape,
		Data:    data,
		Strides: CalculateStrides(shape),
	}
}

// Returns the element at the given index
func (t *Tensor) At(indices ...int) (float32, error) {
	if len(indices) != len(t.Shape) {
		return 0,
			fmt.Errorf(
				"Indices do not match Tensor shape: # of Indices (%d) != Shape (%d)",
				len(indices), len(t.Shape),
			)
	}

	// Basically, an element can be accessed through
	// Data_n = (index_0 * stride_0) + (index_1 * stride_1) + ...
	//          + (index_n * stride_n)

	offset := 0
	for i, idx := range indices {
		if idx < 0 || idx >= int(t.Shape[i]) {
			return 0, fmt.Errorf(
				"Index %d exceeds Axis %d of size %d", idx, i, t.Shape[i],
			)
		}

		offset += (idx * t.Strides[i])
	}

	return t.Data[offset], nil
}

// Broadcasting:
// Axes of a tensor are stretched to adapt it to a given size allowing for
// element-wise operations
func Broadcast(t *Tensor, targetShape []int) (*Tensor, error) {

	targetRank := len(targetShape)
	if len(t.Shape) > targetRank {
		return nil, fmt.Errorf("Cannot broadcast Tensor: Tensor Rank > Target Rank")
	}

	// Step 1: Prepend to Shape and subsequently strides
	broadcastedShape := prepend(t.Shape, targetRank, 1)
	broadcastedStrides := prepend(t.Strides, targetRank, 0)

	// Step 2: Aligning axes
	// >>> Rule 1: One of the axes must be 1 OR both must be equal!
	// >>> Rule 2: Tensors will be broadcasted to the greatest rank
	for i := targetRank - 1; i >= 0; i-- {
		tensorAxis := broadcastedShape[i]
		targetAxis := targetShape[i]

		if !(tensorAxis == 1 || tensorAxis == targetAxis) {
			return nil, fmt.Errorf("Cannot broadcast dimension %d: %d to %d!", i, tensorAxis, targetAxis)
		}

		// Axes are equal, no stretching here
		if tensorAxis != 1 {
			continue
		}

		broadcastedStrides[i] = 0
	}

	return &Tensor{
		Shape:   targetShape,
		Data:    t.Data,
		Strides: broadcastedStrides,
	}, nil
}

// Helper function:
// Prepends n - len(arr) 0's to arr
func prepend(arr []int, n int, prependWith int) []int {
	newArray := make([]int, n)
	for i := 0; i < n-len(arr); i++ {
		newArray[i] = prependWith
	}

	copy(newArray[n-len(arr):], arr)
	return newArray
}
