package layers

import (
	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/nn/topologies"
)

// It's just basically a combination of layers
type TransformerBlock struct {
	Attn        *AttentionLayer
	FeedForward *topologies.Sequential
	Norm1       *LayerNorm
	Norm2       *LayerNorm
}

func NewTransformerBlock(vectorSize, hiddenDimension int) *TransformerBlock {
	ff := topologies.NewSequential()
	ff.Add(NewDenseLayer(vectorSize, hiddenDimension))
	ff.Add(&ReLULayer{})
	ff.Add(NewDenseLayer(hiddenDimension, vectorSize))

	return &TransformerBlock{
		Attn:        NewAttentionLayer(vectorSize),
		FeedForward: ff,
		Norm1:       NewLayerNorm(vectorSize),
		Norm2:       NewLayerNorm(vectorSize),
	}
}

func (t *TransformerBlock) Forward(x *autograd.Variable) *autograd.Variable {
	outputAttention := t.Attn.Forward(x)

	// Computes and normalises residual
	res1 := autograd.Add(x, outputAttention)
	norm1 := t.Norm1.Forward(res1)

	// Feeds normalised residual into sequential model
	outputFF := t.FeedForward.Forward(norm1)

	// Calculates residuals again
	res2 := autograd.Add(norm1, outputFF)
	norm2 := t.Norm2.Forward(res2)

	return norm2
}

// Dumps everything
func (t *TransformerBlock) Parameters() []*autograd.Variable {
	params := []*autograd.Variable{}
	params = append(params, t.Attn.Parameters()...)
	params = append(params, t.FeedForward.Parameters()...)
	params = append(params, t.Norm1.Parameters()...)
	params = append(params, t.Norm2.Parameters()...)

	return params
}
