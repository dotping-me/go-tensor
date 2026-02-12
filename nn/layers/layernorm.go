package layers

import (
	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/tensor"
)

// Basically standardises the outputs of a layer to be numerically stable
// i.e. Mapping outputs on a scale of 0 to 1
type LayerNorm struct {
	Gamma *autograd.Variable
	Beta  *autograd.Variable
	Eps   float32
}

func NewLayerNorm(size int) *LayerNorm {
	gamma := tensor.NewTensor([]int{size}, make([]float32, size))
	beta := tensor.NewTensor([]int{size}, make([]float32, size))
	for i := range size {
		gamma.Data()[i] = 1.0
	}

	return &LayerNorm{
		Gamma: autograd.NewVariable(gamma, true),
		Beta:  autograd.NewVariable(beta, true),
		Eps:   1e-5,
	}
}

func (ln *LayerNorm) Forward(x *autograd.Variable) *autograd.Variable {

	// Normalisation formula:
	// X = (x - mean) / sqrt(variance + eps)
	// y = (gamma * X) + beta

	mean := autograd.Mean(x, -1, true)
	variance := autograd.Variance(x, -1, true)

	// Add scalar to variabce for stability
	eps := autograd.NewVariable(tensor.NewScalar(ln.Eps), false)
	variancePlusEps := autograd.Add(variance, eps)

	// Normalise
	centered := autograd.Sub(x, mean)

	std := autograd.Sqrt(variancePlusEps)
	norm := autograd.Div(centered, std)

	// Maps data onto a scale
	scaled := autograd.Mul(norm, ln.Gamma)
	shifted := autograd.Add(scaled, ln.Beta)

	return shifted
}

func (ln *LayerNorm) Parameters() []*autograd.Variable {
	return []*autograd.Variable{ln.Gamma, ln.Beta}
}
