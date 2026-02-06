// TODO: Implement multi-axis reduction

package tensor

import "fmt"

type onePassReductionOp struct {
	getStartingValue func() float32 // Returns the starting value (i.e For sum -> 0)
	reduce           func(acc, val float32) float32
	finalize         func(acc float32) float32 // Final operation before append
}

func (t *Tensor) reduce(axisIndex int, keepAxis bool, op onePassReductionOp) (*Tensor, error) {
	rank := len(t.shape)
	if axisIndex < 0 {
		axisIndex += rank // Axis normalization
	}

	if axisIndex < 0 || axisIndex >= rank {
		return nil, fmt.Errorf("Cannot sum Axis %d: Rank of Tensor = %d", axisIndex, rank)
	}

	// Determine output shape beforehand
	var outputShape []int
	for i := range rank {
		if i == axisIndex {
			if keepAxis == true {
				outputShape = append(outputShape, 1)
			}

			continue // Removes axis
		}

		outputShape = append(outputShape, t.shape[i])
	}

	// Allocates memory
	outputTensor := NewTensor(
		outputShape, make([]float32, numberOfDataAsPerShape(outputShape)),
	)

	// Starts iterating over tensor data
	w := newWalker(outputTensor)
	for !w.done {
		currentIndex := w.Index()

		// Finds the index of the element to sum because if axis was consumed,
		// there's no referrence that can be used. It's late right now so maybe
		// this isn't making much sense

		indexOfElToSum := make([]int, rank) // Is there a way to optimise this?
		pos := 0                            // I'm just getting the other indices around it
		for i := range rank {
			if (i == axisIndex && keepAxis == true) || (i != axisIndex) {
				indexOfElToSum[i] = currentIndex[pos]
				pos++
			}
		}

		// Starts reduction loop
		acc := op.getStartingValue()

		for i := range t.shape[axisIndex] {
			indexOfElToSum[axisIndex] = i

			el, err := t.At(indexOfElToSum...)
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
		getStartingValue: func() float32 { return -1e38 },
		reduce: func(acc, val float32) float32 {
			if acc > val {
				return val
			}

			return acc
		},

		finalize: func(acc float32) float32 { return acc },
	})
}

func (t *Tensor) Min(axisIndex int, keepAxis bool) (*Tensor, error) {
	return t.reduce(axisIndex, keepAxis, onePassReductionOp{
		getStartingValue: func() float32 { return 1e38 },
		reduce: func(acc, val float32) float32 {
			if acc < val {
				return val
			}

			return acc
		},

		finalize: func(acc float32) float32 { return acc },
	})
}

// func (t *Tensor) LogSumExp(axisIndex int, keepAxis bool) (*Tensor, error) {
// 	maxT, err := t.Max(axisIndex, true)
// 	if err != nil {
// 		return nil, err
// 	}
// }
