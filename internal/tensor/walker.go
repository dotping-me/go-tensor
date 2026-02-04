package tensor

// Just the logic to traverse the tensor because that will be used basically
// everywhere, by everyone and even their grandma!

type Walker struct {
	tensor *Tensor
	index  []int // Simulating the data in N-D space
	offset int
	done   bool
}

func NewWalker(t *Tensor) *Walker {
	return &Walker{
		tensor: t,
		index:  make([]int, len(t.Shape)), // i.e. Rank 3 -> (x, y, z)
		offset: 0,
	}
}

// Gets the current element
func (w *Walker) Value() float32 {
	return w.tensor.Data[w.offset] // 1st value will be at (0, 0, 0, ..., 0)
}

// Moves to the next element in tensor
func (w *Walker) WalkOne() bool {
	if w.done {
		return false
	}

	shape := w.tensor.Shape
	strides := w.tensor.Strides

	// Walks to the next element. Must also update offset if need be
	for axis := len(shape) - 1; axis >= 0; axis-- {

		// Moves to next
		w.index[axis]++
		if w.index[axis] < shape[axis] {
			w.offset += strides[axis]
			return true
		}

		// End of axis reached
		// Start at the first element on the next axis which will be moved to
		// on the next iteration of this loop straight after this
		w.offset -= ((shape[axis] - 1) * strides[axis])
		w.index[axis] = 0
	}

	w.done = true
	return false
}
