package nn

import (
	"fmt"

	"github.com/dotping-me/go-tensor/autograd"
)

// Previous comment here made no sense
// Mais bon, now this is here so I can attach convenient methods to it
type Model struct {
	Root Layer
}

type Layer interface {
	Forward(v *autograd.Variable) *autograd.Variable
	Parameters() []*autograd.Variable
}

func NewModel(root Layer) *Model {
	return &Model{Root: root}
}

func (m *Model) Forward(v *autograd.Variable) *autograd.Variable {
	return m.Root.Forward(v)
}

// Collect all parameters recursively -> Each layer calls their Parameters()
func (m *Model) Parameters() []*autograd.Variable {
	return m.Root.Parameters()
}

// Loading a set amount of parameters
func (m *Model) LoadParameters(params []*autograd.Variable) error {
	existingParams := m.Parameters()
	if len(params) != len(existingParams) {
		return fmt.Errorf("Failed to load new %d parameters: Model uses %d parameters", len(params), len(existingParams))
	}

	// Copies parameters over
	for i := range existingParams {
		existingParams[i].Tensor = params[i].Tensor.Copy()
	}

	return nil
}

// Inference
func (m *Model) Predict(v *autograd.Variable) *autograd.Variable {
	return m.Forward(v)
}
