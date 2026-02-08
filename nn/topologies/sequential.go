package topologies

import (
	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/nn"
)

// TODO: Add more model topologies later
type Sequential struct {
	layers []nn.Layer
}

func NewSequential() *Sequential {
	return &Sequential{}
}

func (s *Sequential) Add(l nn.Layer) {
	s.layers = append(s.layers, l)
}

// Calls Forward on every Layer of the sequential model
func (s *Sequential) Forward(v *autograd.Variable) *autograd.Variable {

	// Accumulates outputs
	allOutputs := v
	for _, l := range s.layers {
		allOutputs = l.Forward(allOutputs)
	}

	return allOutputs
}

// Collects all the variables from all layers iteratively
func (s *Sequential) Parameters() []*autograd.Variable {
	allParams := []*autograd.Variable{}

	for _, l := range s.layers {
		allParams = append(allParams, l.Parameters()...)
	}

	return allParams
}
