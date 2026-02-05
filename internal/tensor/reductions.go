package tensor

import "fmt"

func (t *Tensor) Sum(axisIndex int, keepAxis bool) (*Tensor, error) {
	rank := len(t.shape)

	// TODO: Make Axis normalization universal
	if axisIndex < 0 {
		axisIndex += rank
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

	w := newWalker(outputTensor)
	for !w.done {
		currentIndex := w.Index()

		// Finds the index of the element to sum because if axis was consumed,
		// there's no referrence that can be used. It's late right now so maybe
		// this isn't making much sense

		indexOfElToSum := make([]int, rank)
		pos := 0
		for i := range rank {
			if (i == axisIndex && keepAxis == true) || (i != axisIndex) {
				indexOfElToSum[i] = currentIndex[pos]
				pos++
			}
		}

		// Sums up the elements on that axis
		sum := float32(0)
		for i := range t.shape[axisIndex] {
			indexOfElToSum[axisIndex] = i
			el, err := t.At(indexOfElToSum...) // Je comprend que dalle
			if err != nil {
				return nil, err
			}

			sum += el
		}

		outputTensor.data[w.offset] = sum
		w.WalkOne()
	}

	return outputTensor, nil
}
