// TODO: Implement multi-axis reduction

package tensor

import (
	"fmt"
	"math"
)

type onePassReductionOp struct {
	getStartingValue func() float32 // Returns the starting value (i.e For sum -> 0)
	reduce           func(acc, val float32) float32
	finalize         func(acc float32) float32 // Final operation before append
}

// --------------------------
//          Helpers
// --------------------------

func determineOutputShape(shape []int, axisIndex int, keepAxis bool) []int {
	var outputShape []int
	for i := range len(shape) {
		if i == axisIndex {
			if keepAxis == true {
				outputShape = append(outputShape, 1)
			}

			continue // Removes axis
		}

		outputShape = append(outputShape, shape[i])
	}

	return outputShape
}

// Finds the index of the element to sum because if axis was consumed,
// there's no referrence that can be used. It's late right now so maybe
// this isn't making much sense
func determineConsumedElementIndices(currentIndex []int, axisIndex int, keepAxis bool) []int {
	rank := len(currentIndex)
	if !keepAxis {
		rank++
	}

	indexOfEl := make([]int, rank) // Is there a way to optimise this?
	pos := 0                       // I'm just getting the other indices around it

	for i := range rank {
		if (i == axisIndex && keepAxis == true) || (i != axisIndex) {
			indexOfEl[i] = currentIndex[pos]
			pos++
		}
	}

	return indexOfEl
}

// --------------------------
//   One pass reduction ops
// --------------------------

func (t *Tensor) reduce(axisIndex int, keepAxis bool, op onePassReductionOp) (*Tensor, error) {
	rank := len(t.shape)
	if axisIndex < 0 {
		axisIndex += rank // Axis normalization
	}

	if axisIndex < 0 || axisIndex >= rank {
		return nil, fmt.Errorf("Cannot sum Axis %d: Rank of Tensor = %d", axisIndex, rank)
	}

	// Allocates memory
	outputShape := determineOutputShape(t.shape, axisIndex, keepAxis)
	outputTensor := NewTensor(
		outputShape, make([]float32, numberOfDataAsPerShape(outputShape)),
	)

	// Starts iterating over tensor data
	w := newWalker(outputTensor)
	for !w.done {
		currentIndex := w.Index()
		indexOfEl := determineConsumedElementIndices(currentIndex, axisIndex, keepAxis)

		// Starts reduction loop
		acc := op.getStartingValue()
		for i := range t.shape[axisIndex] {
			indexOfEl[axisIndex] = i

			el, err := t.At(indexOfEl...)
			if err != nil {
				return nil, err
			}

			acc = op.reduce(acc, el)
		}

		outputTensor.data[w.offset] = op.finalize(acc)
		w.WalkOne()
	}

	return outputTensor, nil
}

func (t *Tensor) Sum(axisIndex int, keepAxis bool) (*Tensor, error) {
	return t.reduce(axisIndex, keepAxis, onePassReductionOp{
		getStartingValue: func() float32 { return 0 },
		reduce:           func(acc, val float32) float32 { return acc + val },
		finalize:         func(acc float32) float32 { return acc },
	})
}

func (t *Tensor) Mean(axisIndex int, keepAxis bool) (*Tensor, error) {
	return t.reduce(axisIndex, keepAxis, onePassReductionOp{
		getStartingValue: func() float32 { return 0 },
		reduce:           func(acc, val float32) float32 { return acc + val },

		finalize: func(acc float32) float32 {
			return acc / float32(t.shape[axisIndex])
		},
	})
}

func (t *Tensor) Max(axisIndex int, keepAxis bool) (*Tensor, error) {
	return t.reduce(axisIndex, keepAxis, onePassReductionOp{
		getStartingValue: func() float32 { return t.data[0] },
		reduce: func(acc, val float32) float32 {
			if val > acc {
				return val
			}

			return acc
		},

		finalize: func(acc float32) float32 { return acc },
	})
}

func (t *Tensor) Min(axisIndex int, keepAxis bool) (*Tensor, error) {
	return t.reduce(axisIndex, keepAxis, onePassReductionOp{
		getStartingValue: func() float32 { return t.data[0] },
		reduce: func(acc, val float32) float32 {
			if val < acc {
				return val
			}

			return acc
		},

		finalize: func(acc float32) float32 { return acc },
	})
}

// --------------------------------------------
//   Reduction Ops with more than just 1 pass
// --------------------------------------------

// I may be writing more lines but at least I'm not doing more calls, I hope...
// TODO: Maybe there's a possibility of making maybe like
//       an N-pass func struct

// Finds the index of the max element per row
func (t *Tensor) ArgMax(axisIndex int, keepAxis bool) (*Tensor, error) {
	rank := len(t.shape)
	if axisIndex < 0 {
		axisIndex += rank
	}

	if axisIndex < 0 || axisIndex >= rank {
		return nil, fmt.Errorf("Cannot sum Axis %d: Rank of Tensor = %d", axisIndex, rank)
	}

	// Allocates memory
	outputShape := determineOutputShape(t.shape, axisIndex, keepAxis)
	outputTensor := NewTensor(
		outputShape, make([]float32, numberOfDataAsPerShape(outputShape)),
	)

	// Starts iterating over tensor data
	w := newWalker(outputTensor)
	for !w.done {
		currentIndex := w.Index()
		indexOfEl := determineConsumedElementIndices(currentIndex, axisIndex, keepAxis)

		// Starts reduction loop
		maxIndex := 0

		indexOfEl[axisIndex] = maxIndex
		maxEl, err := t.At(indexOfEl...)
		if err != nil {
			return nil, err
		}

		for i := range t.shape[axisIndex] {
			indexOfEl[axisIndex] = i

			el, err := t.At(indexOfEl...)
			if err != nil {
				return nil, err
			}

			if el > maxEl {
				maxEl = el
				maxIndex = i
			}
		}

		outputTensor.data[w.offset] = float32(maxIndex)
		w.WalkOne()
	}

	return outputTensor, nil
}

// Finds the index of the min element per row
func (t *Tensor) ArgMin(axisIndex int, keepAxis bool) (*Tensor, error) {
	rank := len(t.shape)
	if axisIndex < 0 {
		axisIndex += rank
	}

	if axisIndex < 0 || axisIndex >= rank {
		return nil, fmt.Errorf("Cannot sum Axis %d: Rank of Tensor = %d", axisIndex, rank)
	}

	// Allocates memory
	outputShape := determineOutputShape(t.shape, axisIndex, keepAxis)
	outputTensor := NewTensor(
		outputShape, make([]float32, numberOfDataAsPerShape(outputShape)),
	)

	// Starts iterating over tensor data
	w := newWalker(outputTensor)
	for !w.done {
		currentIndex := w.Index()
		indexOfEl := determineConsumedElementIndices(currentIndex, axisIndex, keepAxis)

		// Starts reduction loop
		minIndex := 0

		indexOfEl[axisIndex] = minIndex
		minEl, err := t.At(indexOfEl...)
		if err != nil {
			return nil, err
		}

		for i := range t.shape[axisIndex] {
			indexOfEl[axisIndex] = i

			el, err := t.At(indexOfEl...)
			if err != nil {
				return nil, err
			}

			if el < minEl {
				minEl = el
				minIndex = i
			}
		}

		outputTensor.data[w.offset] = float32(minIndex)
		w.WalkOne()
	}

	return outputTensor, nil
}

// 3-Pass reduction operation!
// Basically does: log( sum( exp( x - max(x) ) ) ) + max(x) (TFFFF????)
func (t *Tensor) LogSumExp(axisIndex int, keepAxis bool) (*Tensor, error) {
	rank := len(t.shape)
	if axisIndex < 0 {
		axisIndex += rank
	}

	if axisIndex < 0 || axisIndex >= rank {
		return nil, fmt.Errorf("Cannot sum Axis %d: Rank of Tensor = %d", axisIndex, rank)
	}

	// Allocates memory
	outputShape := determineOutputShape(t.shape, axisIndex, keepAxis)
	outputTensor := NewTensor(
		outputShape, make([]float32, numberOfDataAsPerShape(outputShape)),
	)

	// Starts iterating over tensor data
	w := newWalker(outputTensor)
	for !w.done {
		currentIndex := w.Index()
		indexOfEl := determineConsumedElementIndices(currentIndex, axisIndex, keepAxis)

		// Starts reduction loop

		// Step 1: Find the maximum element
		indexOfEl[axisIndex] = 0
		maxEl, err := t.At(indexOfEl...)
		if err != nil {
			return nil, err
		}

		for i := range t.shape[axisIndex] {
			indexOfEl[axisIndex] = i

			el, err := t.At(indexOfEl...)
			if err != nil { // TODO: Would removing these checks improve performance?
				return nil, err
			}

			if el > maxEl {
				maxEl = el
			}
		}

		// Step 2: Sum the exponential of that max value
		sumExp := float32(0)
		for i := range t.shape[axisIndex] {
			indexOfEl[axisIndex] = i

			el, err := t.At(indexOfEl...)
			if err != nil {
				return nil, err
			}

			sumExp += float32(math.Exp(float64(el - maxEl)))
		}

		// Step 3 here!
		outputTensor.data[w.offset] = float32(math.Log(float64(sumExp))) + maxEl
		w.WalkOne()
	}

	return outputTensor, nil
}

// Variance: Highschool maths
// = mean((x - X)**2)
func (t *Tensor) Variance(axisIndex int, keepAxis bool) (*Tensor, error) {
	mean, err := t.Mean(axisIndex, true) // Finds the mean
	if err != nil {
		return nil, err
	}

	rank := len(t.shape)
	if axisIndex < 0 {
		axisIndex += rank
	}

	if axisIndex < 0 || axisIndex >= rank {
		return nil, fmt.Errorf("Cannot sum Axis %d: Rank of Tensor = %d", axisIndex, rank)
	}

	// Allocates memory
	outputShape := determineOutputShape(t.shape, axisIndex, keepAxis)
	outputTensor := NewTensor(
		outputShape, make([]float32, numberOfDataAsPerShape(outputShape)),
	)

	// Starts iterating over tensor data
	w := newWalker(outputTensor)
	for !w.done {
		currentIndex := w.Index()
		indexOfEl := determineConsumedElementIndices(currentIndex, axisIndex, keepAxis)

		// Starts reduction loop
		m, err := mean.At(indexOfEl...)
		if err != nil {
			return nil, err
		}

		sum := float32(0)
		for i := range t.shape[axisIndex] {
			indexOfEl[axisIndex] = i

			el, err := t.At(indexOfEl...)
			if err != nil {
				return nil, err
			}

			diff := el - m
			sum += (diff * diff)
		}

		outputTensor.data[w.offset] = sum / float32(t.shape[axisIndex])
		w.WalkOne()
	}

	return outputTensor, nil
}

// Standard deviation is just the square root of variance
func (t *Tensor) StandardDeviation(axisIndex int, keepAxis bool) (*Tensor, error) {
	variance, err := t.Variance(axisIndex, keepAxis)
	if err != nil {
		return nil, err
	}

	for i := range variance.data {
		variance.data[i] = float32(math.Sqrt(float64(variance.data[i])))
	}

	return variance, nil
}

// It looks like a fever dream
// Basically: softmax(x_i) = e**x_i / (∑_j (e**x_i))
func Softmax(x *Tensor, axisIndex int) (*Tensor, error) {
	lse, err := x.LogSumExp(axisIndex, true)
	if err != nil {
		return nil, err
	}

	diff, err := Sub(x, lse)
	if err != nil {
		return nil, err
	}

	exps, err := Exp(diff)
	if err != nil {
		return nil, err
	}

	sum, err := exps.Sum(axisIndex, true)
	if err != nil {
		return nil, err
	}

	return Div(exps, sum)
}
