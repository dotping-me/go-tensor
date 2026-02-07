package utils

import "github.com/dotping-me/go-tensor/tensor"

func Batch(X, Y *tensor.Tensor, batchSize int) [][2]*tensor.Tensor {
	var batches [][2]*tensor.Tensor

	N := len(X.Data()) / X.Shape()[1]
	for i := 0; i < N; i += batchSize {
		end := i + batchSize
		if end > N {
			end = N
		}

		xBatch, _ := X.SlicePerAxis(tensor.Slice{Start: 0, End: end, Step: 1})
		yBatch, _ := Y.SlicePerAxis(tensor.Slice{Start: 0, End: end, Step: 1})

		batches = append(batches, [2]*tensor.Tensor{xBatch, yBatch})
	}
	return batches
}
