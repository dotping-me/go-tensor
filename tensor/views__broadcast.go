package tensor

import "fmt"

// TODO: Make some of these functions into Tensor methods

// Helper function: prependWiths n - len(arr) 0's to arr
func prependWith(arr []int, n int, v int) []int {
	newArray := make([]int, n)
	for i := 0; i < n-len(arr); i++ {
		newArray[i] = v
	}

	copy(newArray[n-len(arr):], arr)
	return newArray
}

// Broadcasting:
// Axes of a tensor are stretched to adapt it to a given size allowing for
// element-wise operations
func findBroadcastShape(shape1 []int, shape2 []int) ([]int, error) {
	lengthA := len(shape1)
	lengthB := len(shape2)

	maxRank := max(lengthA, lengthB)
	outputShape := prependWith([]int{}, maxRank, 1)
	for i := range maxRank {
		if i > lengthA-1 && i <= lengthB-1 {
			outputShape[i] = shape2[i]
			continue

		} else if i > lengthB-1 && i <= lengthA-1 {
			outputShape[i] = shape1[i]
			continue

		} else if shape1[i] != 1 && shape2[i] != 1 && shape1[i] != shape2[i] {
			return nil, fmt.Errorf(
				"Cannot broadcast Shapes %v and %v at Axis %v!", shape1, shape2, i,
			)
		}

		outputShape[i] = max(shape1[i], shape2[i])
	}

	return outputShape, nil
}

func Broadcast(t *Tensor, targetShape []int) (*Tensor, error) {
	targetRank := len(targetShape)
	if len(t.shape) > targetRank {
		return nil, fmt.Errorf("Cannot broadcast Tensor: Tensor Rank > Target Rank")
	}

	// NOTE: Use FindBroadcastShape() to optimize this here?

	// Step 1: prependWith to Shape and subsequently strides
	broadcastedShape := prependWith(t.shape, targetRank, 1)
	broadcastedStrides := prependWith(t.strides, targetRank, 0)

	// Step 2: Aligning axes
	// >>> Rule 1: One of the axes must be 1 OR both must be equal!
	// >>> Rule 2: Tensors will be broadcasted to the greatest rank
	for i := targetRank - 1; i >= 0; i-- {
		tensorAxis := broadcastedShape[i]
		targetAxis := targetShape[i]

		if tensorAxis != 1 && targetAxis != 1 && tensorAxis != targetAxis {
			return nil, fmt.Errorf("Cannot broadcast dimension %d: %d to %d!", i, tensorAxis, targetAxis)
		}

		// Axis of original tensor dominates -> Unchanged
		if tensorAxis > targetAxis {
			targetShape[i] = tensorAxis
		}

		// Axes are equal, no stretching here
		if tensorAxis != 1 {
			continue
		}

		broadcastedStrides[i] = 0
	}

	return &Tensor{
		shape:   targetShape,
		data:    t.data,
		strides: broadcastedStrides,
	}, nil
}

// Helper for autograd
// Basically undoes the broadcasting and returns the tensor to its original shape
// before broadcasting, however data is changed
func BroadcastBackward(t, originalT *Tensor) (*Tensor, error) {
	outputTensor := t
	var err error

	// Align the ranks (prepends axes)
	if len(outputTensor.shape) < len(originalT.shape) {
		newShape := append([]int{1}, t.shape...)
		outputTensor, err = Reshape(outputTensor, newShape)
		if err != nil {
			return nil, err
		}
	}

	// Reduce broadcasted axes
	for i := range originalT.shape {
		if originalT.shape[i] == 1 && outputTensor.shape[i] > 1 {
			outputTensor, err = outputTensor.Sum(i, true)
			if err != nil {
				return nil, err
			}
		}
	}

	return outputTensor, nil
}
