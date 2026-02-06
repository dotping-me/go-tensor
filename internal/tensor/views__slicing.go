package tensor

import (
	"fmt"
)

type Slice struct {
	Start int
	End   int
	Step  int
}

// Selects the full axis
func All() Slice {
	return Slice{Start: 0, End: -1, Step: 1}
}

// NOTE: Important distinction -> A slice IS a tensor!
func (t *Tensor) SlicePerAxis(slices ...Slice) (*Tensor, error) {
	rank := len(t.shape)
	if len(slices) > rank {
		return nil, fmt.Errorf("# of Slices exceeds Tensor rank: %d > %d", len(slices), rank)
	}

	newShape := make([]int, rank)
	newStrides := make([]int, rank)
	sliceOffset := t.sliceOffset

	for axis := range rank {
		start := 0
		end := t.shape[axis] // Default values
		step := 1

		// Since the number of arguments corresponds to the axis index,
		// Check for whether a slice was provided for this axis
		// Otherwise, use default settings
		if axis < len(slices) {
			s := slices[axis]
			start = s.Start
			step = s.Step

			if s.End == -1 {
				end = t.shape[axis]

			} else {
				end = s.End
			}

			// Validates slice
			if start < 0 || start >= t.shape[axis] || start >= end || step <= 0 {
				return nil, fmt.Errorf("Slice parameters are invalid: %v!", s)
			}

		}

		// Calculate the size of that axis for the slice
		axisSize := (end - start + (step - 1)) / step // (step - 1) if ever step > 1
		sliceOffset += start * t.strides[axis]

		newShape[axis] = axisSize
		newStrides[axis] = t.strides[axis] * step
	}

	return &Tensor{
		shape:       newShape,
		data:        t.data,
		strides:     newStrides,
		sliceOffset: sliceOffset,
	}, nil
}

// NOTE: Helper function for batched matrix multiplication!
func (t *Tensor) Get2DSliceAtParentIndex(parentIndex []int) (*Tensor, error) {
	slices := make([]Slice, len(t.shape))

	for i := range t.shape { // Because index is not a fixed shape/rank
		if i < len(parentIndex) {
			slices[i] = Slice{
				Start: parentIndex[i],
				End:   parentIndex[i] + 1, // Selecting just 1 axis
				Step:  1,
			}

			continue
		}

		slices[i] = All() // Select 2D matrices
	}

	batch, err := t.SlicePerAxis(slices...)
	if err != nil {
		return nil, err
	}

	// TODO: It might be a good idea to make that its own function
	// Drops the prepending axes because they're just 1's and I need
	// 2D matrices
	bs := batch.shape
	if len(bs) > 2 {
		batch.shape = bs[len(bs)-2:]
		batch.strides = batch.strides[len(bs)-2:]
	}

	return batch, nil
}
