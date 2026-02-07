package layers

import "github.com/dotping-me/go-tensor/autograd"

// ReLU activation function
// Basically it becomes 1 if output > 0, else 0

type ReLULayer struct{}

func (r *ReLULayer) Forward(v *autograd.Variable) *autograd.Variable {
	return autograd.ReLU(v)
}

func (r *ReLULayer) Parameters() []*autograd.Variable {
	return nil
}
