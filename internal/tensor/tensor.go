// TODO: Modularise everything properly later

package tensor

import (
	"fmt"
	"strings"
)

// TODO: Abstract this
type Tensor struct {
	shape       []int
	data        []float32
	strides     []int
	sliceOffset int
}

// Constructor which calculates strides on instantiation
func NewTensor(shape []int, data []float32) *Tensor {

	// TODO: Shape to Data validation

	return &Tensor{
		shape:       shape,
		data:        data,
		strides:     calculateStrides(shape),
		sliceOffset: 0,
	}
}

// TODO: Fix this!!!
// String representation
func (t *Tensor) String() string {
	rank := len(t.shape)
	if rank == 0 {
		return fmt.Sprintf("%v", t.data[0])
	}

	var sb strings.Builder
	w := newWalker(t)
	fmt.Fprintf(&sb, "%s", fmt.Sprintf("T (Shape: %v): ", t.shape))

	// Write opening brackets
	for range rank {
		fmt.Fprintf(&sb, "[")
	}

	for {
		if w.done {
			break
		}

		fmt.Fprintf(&sb, "%f", w.Value())

		// Check if axis was changed
		currentIndex := w.Index()
		for i := rank - 1; i >= 0; i-- {
			if currentIndex[i] == t.shape[i]-1 && i != 0 {
				fmt.Fprintf(&sb, "] [")

			} else if i != 0 {
				fmt.Fprintf(&sb, " ")
			}
		}

		w.WalkOne()
	}

	// Truncates extra " ["
	s := sb.String()
	s = s[:len(s)-2]

	// Last element reached, close all open brackets
	for range rank - 1 {
		s = s + "]"
	}

	return s
}

func (t *Tensor) Data() []float32 {
	return t.data
}

// Calculates the strides for a tensor
// A stride is basically the # of elements that needs to be moved per axis to
// access the next batch
func calculateStrides(shape []int) []int {
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

// Finds the number of data elements according to shape
func numberOfDataAsPerShape(shape []int) int {
	total := 1
	for _, axis := range shape {
		total *= axis
	}

	return total
}

// Returns the element at the given index
func (t *Tensor) At(indices ...int) (float32, error) {
	if len(indices) != len(t.shape) {
		return 0,
			fmt.Errorf(
				"Indices do not match Tensor shape: # of Indices (%d) != Shape (%d)",
				len(indices), len(t.shape),
			)
	}

	// Basically, an element can be accessed through
	// Data_n = (index_0 * stride_0) + (index_1 * stride_1) + ...
	//          + (index_n * stride_n)

	offset := 0
	for i, idx := range indices {

		// NOTE: Ignore broadcasted axes
		if (idx < 0 || idx >= int(t.shape[i])) && t.strides[i] != 0 {
			return 0, fmt.Errorf(
				"Index %d exceeds Axis %d of size %d", idx, i, t.shape[i],
			)
		}

		offset += (idx * t.strides[i])
	}

	return t.data[t.sliceOffset+offset], nil
}

// Modifies a tensor so that it takes a new shape
func Reshape(t *Tensor, newShape []int) (*Tensor, error) {
	newNumberOfData := numberOfDataAsPerShape(newShape)
	if newNumberOfData != numberOfDataAsPerShape(t.shape) {
		return nil, fmt.Errorf(
			"Cannot reshape Tensor of Shape %v to %v: # of elements do not match",
			t.shape, newShape,
		)
	}

	return NewTensor(newShape, t.data), nil
}

// Destructures axes and data are mapped onto a shape of [1]
func Flatten(t *Tensor) *Tensor {
	return &Tensor{
		shape:   []int{numberOfDataAsPerShape(t.shape)},
		data:    t.data,
		strides: []int{1},
	}
}

// TODO: Maybe I'll implement something like Pytorch's expandDims later

// Swaps around the tensor's shape and strides to transpose same data over
// different axes
func Transpose(t *Tensor, shiftToThisAxisIndex []int) (*Tensor, error) {

	rank := len(t.shape)
	if rank != len(shiftToThisAxisIndex) {
		return nil, fmt.Errorf(
			"Cannot transpose Shape %v onto %v: Unequal length", t.shape, shiftToThisAxisIndex,
		)
	}

	// Shift axes
	newShape := make([]int, len(t.shape))
	for i, axisIndex := range shiftToThisAxisIndex {
		if axisIndex < 0 || axisIndex >= rank {
			return nil, fmt.Errorf(
				"Transpose error: Axis index (%d) > rank (%d)!",
				axisIndex+1, rank,
			)
		}

		newShape[i] = t.shape[axisIndex]
	}

	return NewTensor(newShape, t.data), nil
}
