package tensor

// Just the logic to traverse the tensor because that will be used basically
// everywhere, by everyone and even their grandma!

type walker struct {
	tensor *Tensor
	index  []int // Simulating the data in N-D space
	offset int
	done   bool
}

// Constructor
func newWalker(t *Tensor) *walker {
	return &walker{
		tensor: t,
		index:  make([]int, len(t.shape)), // i.e. Rank 3 -> (x, y, z)
		offset: 0,
	}
}

func (w *walker) Index() []int {
	return w.index // Read-only copy of data
}

// Gets the current element
func (w *walker) Value() float32 {
	return w.tensor.data[w.tensor.sliceOffset+w.offset] // 1st value will be at (0, 0, 0, ..., 0)
}

// Moves to the next element in tensor
func (w *walker) WalkOne() bool {
	if w.done {
		return false
	}

	shape := w.tensor.shape
	strides := w.tensor.strides

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

// ------------------------------------
//  Utility to just walk along a shape
//  used for the batched matmul func()
// ------------------------------------

// Because I think it's better than just creating another tensor
// just to walk indices

type shapeWalker struct {
	shape   []int
	strides []int
	index   []int
	offset  int
	done    bool
}

func newShapeWalker(shape []int) *shapeWalker {
	return &shapeWalker{
		shape:  shape,
		index:  make([]int, len(shape)),
		offset: 0,
	}
}

func (sw *shapeWalker) Index() []int {
	return sw.index
}

func (sw *shapeWalker) WalkOne() bool {
	if sw.done {
		return false
	}

	shape := sw.shape
	for axis := len(shape) - 1; axis >= 0; axis-- {

		// Moves to next
		sw.index[axis]++
		if sw.index[axis] < shape[axis] {
			return true
		}

		// End of axis reached
		// Start at the first element on the next axis which will be moved to
		// on the next iteration of this loop straight after this
		sw.index[axis] = 0
	}

	sw.done = true
	return false
}
