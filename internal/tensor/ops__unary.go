// Arithmetic operations involving just one single tensor

package tensor

import "math"

func elwiseOpsWithSingleTensor(
	t *Tensor,
	callback func(float32) float32, // TODO: Maybe return an error for sanity?
) (*Tensor, error) {
	numOfUniqueData := len(t.data)

	// Just needs to iterate over data, right? Because shape remains unchanged
	newData := make([]float32, numOfUniqueData)
	for i := range numOfUniqueData {
		newData[i] = callback(t.data[i])
	}

	return NewTensor(t.shape, newData), nil
}

func (t *Tensor) Exp() (*Tensor, error) {
	return elwiseOpsWithSingleTensor(
		t, func(x float32) float32 {
			return float32(math.Exp(float64(x)))
		},
	)
}

func (t *Tensor) Log() (*Tensor, error) {
	return elwiseOpsWithSingleTensor(
		t, func(x float32) float32 {
			return float32(math.Log(float64(x)))
		},
	)
}

func (t *Tensor) Neg() (*Tensor, error) {
	return elwiseOpsWithSingleTensor(
		t, func(x float32) float32 {
			return -x
		},
	)
}
