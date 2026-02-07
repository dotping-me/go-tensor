package autograd

import "github.com/dotping-me/go-tensor/tensor"

type Variable struct {
	Tensor       *tensor.Tensor
	Grad         *tensor.Tensor // Created during backward pass
	Parents      []*Variable
	BackwardFunc func()
	RequiresGrad bool
}

// Constructor
func NewVariable(t *tensor.Tensor, requiresGrad bool) *Variable {
	return &Variable{
		Tensor:       t,
		RequiresGrad: requiresGrad,
	}
}

func (v *Variable) SetGradToNil() {
	v.Grad = nil
}

func (v *Variable) setGradToZeros() {
	if v != nil {
		v.Grad = tensor.NewConstantValTensor(v.Grad, 0)
	}
}

// Iterates through a variable, finding all its parents and those parents' parents
// and so on...
func FindParentTensorsRecursively(v *Variable) []*Variable {
	visitedTensors := map[*Variable]bool{}
	var traversalTree []*Variable

	// Function is declared here so it has access to local variable visitedTensors
	var visit func(*Variable)
	visit = func(visiting *Variable) {
		if visitedTensors[visiting] {
			return
		}

		// Marks tensor as visited and proceeds to visit its parents
		visitedTensors[visiting] = true
		for _, p := range visiting.Parents {
			visit(p)
		}

		// Tensors will be appended like into a stack this way as the recursive
		// call needs to be concluded first
		traversalTree = append(traversalTree, visiting)
	}

	visit(v)             // Starts recursive loop
	return traversalTree // Exits if there's nothing to visit
}

func SumGrad(dst **tensor.Tensor, src *tensor.Tensor) {

	// Assigns the reference to the copy of the tensor to the memory address
	if *dst == nil {
		*dst = src.Copy()
		return
	}

	*dst, _ = (*dst).Add(src) // Edits the actual memory
}

// This is what it's all about!!
func Backward(v *Variable) {
	traversalTree := FindParentTensorsRecursively(v)
	v.Grad = tensor.NewConstantValTensor(v.Tensor, 1)

	// Traverses tree backwards (starting with this v variable, then its parents)
	for i := len(traversalTree) - 1; i >= 0; i-- {
		node := traversalTree[i]

		if node.BackwardFunc != nil {
			node.BackwardFunc()
		}
	}
}
