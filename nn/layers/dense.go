package layers

import (
	"fmt"

	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/tensor"
)

type DenseLayer struct {
	W *autograd.Variable
	b *autograd.Variable
}

// Basically a collection of neurons
func NewDenseLayer(numberOfInputs, numberOfOutputs int) *DenseLayer {
	W := autograd.NewVariable(
		tensor.NewRandomTensor([]int{numberOfInputs, numberOfOutputs}), // Starts with a random weight
		true,
	)

	b := autograd.NewVariable(
		tensor.NewTensor([]int{1, numberOfOutputs}, make([]float32, numberOfOutputs)), // Starts with a random weight
		true,
	)

	return &DenseLayer{W: W, b: b}
}

// Basically does neuron stuff; y = Wx + b, for each input
func (d *DenseLayer) Forward(v *autograd.Variable) *autograd.Variable {
	fmt.Println("Dense Forward...")
	y := autograd.Matrix2dMul(v, d.W)
	return autograd.Add(y, d.b)
}

func (d *DenseLayer) Parameters() []*autograd.Variable {
	return []*autograd.Variable{d.W, d.b} // Dumps its values
}
