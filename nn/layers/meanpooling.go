package layers

import (
	"fmt"

	"github.com/dotping-me/go-tensor/autograd"
)

type MeanPoolingLayer struct{}

func NewMeanPoolingLayer(seqLen int) *MeanPoolingLayer {
	return &MeanPoolingLayer{}
}

func (m *MeanPoolingLayer) Forward(x *autograd.Variable) *autograd.Variable {
	out := autograd.Mean(x, 1, false)
	fmt.Println("Mean:", out.Tensor)
	return out
}

func (m *MeanPoolingLayer) Parameters() []*autograd.Variable {
	return nil
}
