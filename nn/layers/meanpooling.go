package layers

import (
	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/tensor"
)

type MeanPoolingLayer struct {
	SeqLen int
}

func NewMeanPoolingLayer(seqLen int) *MeanPoolingLayer {
	return &MeanPoolingLayer{SeqLen: seqLen}
}

func (m *MeanPoolingLayer) Forward(x *autograd.Variable) *autograd.Variable {

	shape := x.Tensor.Shape()
	totalTokens := shape[0]
	dim := shape[1]

	batchSize := totalTokens / m.SeqLen

	outData := make([]float32, batchSize*dim)

	for b := 0; b < batchSize; b++ {
		for s := 0; s < m.SeqLen; s++ {
			for d := 0; d < dim; d++ {

				inputIndex :=
					(b*m.SeqLen+s)*dim + d

				outputIndex :=
					b*dim + d

				outData[outputIndex] +=
					x.Tensor.Data()[inputIndex] / float32(m.SeqLen)
			}
		}
	}

	t := tensor.NewTensor([]int{batchSize, dim}, outData)

	return autograd.NewVariable(t, false)
}

func (m *MeanPoolingLayer) Parameters() []*autograd.Variable {
	return nil
}
