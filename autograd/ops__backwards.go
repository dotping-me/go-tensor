// Extending tensor operations, adding a callback function to undo their effects
// during backwards pass

package autograd

import (
	"log"

	"github.com/dotping-me/go-tensor/tensor"
)

// TODO: Maybe these can be made into a modular fashion like Tensor elementwise ops
// TODO: Cleanup this file

// -----------------------------------
//   Backwards Func() for Binary ops
// -----------------------------------

func Add(a, b *Variable) *Variable {
	outputTensor, err := a.Tensor.Add(b.Tensor)
	if err != nil {
		log.Fatal(err) // returning errors would just be a mess
	}

	outputVariable := &Variable{
		Tensor:       outputTensor,
		Parents:      []*Variable{a, b},
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
	}

	// Initialises that variable's backwards function here
	if outputVariable.RequiresGrad {
		outputVariable.BackwardFunc = func() {

			// Computes gradients
			if a.RequiresGrad {
				SumGrad(&a.Grad, outputVariable.Grad)
			}

			// NOTE: Biases will ALSO have to be summed back here
			//       Usually this is where all that will happen I thinnk
			//       because Y = wX + B -> + B is always the second parent
			if b.RequiresGrad {
				grad := outputVariable.Grad

				if !grad.IsShapeEqualTo(b.Tensor.Shape()) {
					grad, _ = tensor.BroadcastBackward(grad, b.Tensor)
				}

				SumGrad(&b.Grad, grad)
			}

		}
	}

	return outputVariable
}

func Sub(a, b *Variable) *Variable {
	outputTensor, err := a.Tensor.Sub(b.Tensor)
	if err != nil {
		log.Fatal(err)
	}

	outputVariable := &Variable{
		Tensor:       outputTensor,
		Parents:      []*Variable{a, b},
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
	}

	// Initialises that variable's backwards function here
	if outputVariable.RequiresGrad {
		outputVariable.BackwardFunc = func() {

			// Computes gradients
			if a.RequiresGrad {
				SumGrad(&a.Grad, outputVariable.Grad)
			}

			if b.RequiresGrad {
				negGrad, _ := outputVariable.Grad.Neg()
				SumGrad(&b.Grad, negGrad)
			}

		}
	}

	return outputVariable
}

func Mul(a, b *Variable) *Variable {
	outputTensor, err := a.Tensor.Mul(b.Tensor)
	if err != nil {
		log.Fatal(err)
	}

	outputVariable := &Variable{
		Tensor:       outputTensor,
		Parents:      []*Variable{a, b},
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
	}

	// Initialises that variable's backwards function here
	if outputVariable.RequiresGrad {
		outputVariable.BackwardFunc = func() {

			// Computes gradients
			if a.RequiresGrad {
				gradA, err := outputVariable.Grad.Mul(b.Tensor)
				if err != nil {
					log.Fatal(err) // TODO: Maybe these checks can be removed for optimisation
				}

				SumGrad(&a.Grad, gradA)
			}

			if b.RequiresGrad {
				gradB, err := outputVariable.Grad.Mul(a.Tensor)
				if err != nil {
					log.Fatal(err)
				}

				SumGrad(&b.Grad, gradB)
			}

		}
	}

	return outputVariable
}

// -----------------------------------
//   Backwards Func() for Matrix ops
// -----------------------------------

func Matrix2dMul(a, b *Variable) *Variable {
	outputTensor, err := a.Tensor.Matrix2DMul(b.Tensor)
	if err != nil {
		log.Fatal(err)
	}

	outputVariable := &Variable{
		Tensor:       outputTensor,
		Parents:      []*Variable{a, b},
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
	}

	// Initialises that variable's backwards function here
	if outputVariable.RequiresGrad {
		outputVariable.BackwardFunc = func() {

			// Computes gradients
			if a.RequiresGrad {
				tB, err := b.Tensor.Transpose2D()
				if err != nil {
					log.Fatal(err)
				}

				gradA, err := outputVariable.Grad.Matrix2DMul(tB)
				if err != nil {
					log.Fatal(err)
				}

				SumGrad(&a.Grad, gradA)
			}

			if b.RequiresGrad {
				tA, err := a.Tensor.Transpose2D()
				if err != nil {
					log.Fatal(err)
				}

				gradB, err := tA.Matrix2DMul(outputVariable.Grad)
				if err != nil {
					log.Fatal(err)
				}

				SumGrad(&b.Grad, gradB)
			}

		}
	}

	return outputVariable
}

// --------------------------------------
//   Backwards Func() for Reduction ops
// --------------------------------------

func Sum(a *Variable, axisIndex int, keepAxis bool) *Variable {
	outputTensor, err := a.Tensor.Sum(axisIndex, keepAxis)
	if err != nil {
		panic(err)
	}

	outputVariable := &Variable{
		Tensor:       outputTensor,
		Parents:      []*Variable{a},
		RequiresGrad: a.RequiresGrad,
	}

	if outputVariable.RequiresGrad {
		outputVariable.BackwardFunc = func() {
			grad := outputVariable.Grad
			if !keepAxis {
				grad, _ = grad.ExpandDims(axisIndex)
			}

			grad, err := tensor.Broadcast(grad, a.Tensor.Shape())
			if err != nil {
				log.Fatal(err)
			}

			SumGrad(&a.Grad, grad)
		}
	}

	return outputVariable
}

// -------------------------------------------
//   Backwards Func() for Linear Algebra ops
// -------------------------------------------

func Softmax(a *Variable, axisIndex int) *Variable {
	outputTensor, err := a.Tensor.Softmax(axisIndex)
	if err != nil {
		panic(err)
	}

	outputVariable := &Variable{
		Tensor:       outputTensor,
		Parents:      []*Variable{a},
		RequiresGrad: a.RequiresGrad,
	}

	// Jacobian-vector product
	// A bunch of magic and fairy dust to undo softmax

	if outputVariable.RequiresGrad {
		outputVariable.BackwardFunc = func() {
			tmp, _ := outputVariable.Grad.Mul(outputVariable.Tensor)
			sum, _ := tmp.Sum(axisIndex, true)
			diff, _ := outputVariable.Grad.Sub(sum)
			grad, _ := diff.Mul(outputVariable.Tensor)
			SumGrad(&a.Grad, grad)
		}
	}

	return outputVariable
}

// If softmax was hard, then you do not want to know what this is
func CrossEntropy(x, y *Variable) *Variable {

	// Let's worry about the errors another time
	probs := Softmax(x, 1)
	logp, _ := probs.Tensor.Log()
	masked, _ := logp.Mul(y.Tensor)
	sum, _ := masked.Sum(1, false)
	lossTensor, _ := sum.Neg()

	outputVariable := &Variable{
		Tensor:       lossTensor,
		Parents:      []*Variable{x},
		RequiresGrad: x.RequiresGrad,
	}

	if outputVariable.RequiresGrad {
		outputVariable.BackwardFunc = func() {
			diff, _ := probs.Tensor.Sub(y.Tensor)
			SumGrad(&x.Grad, diff)
		}
	}

	return outputVariable
}

// -----------------------------------------
//   Backwards Func() for Activation funcs
// -----------------------------------------

func ReLU(x *Variable) *Variable {
	outputTensor := x.Tensor.Copy()
	outputData := outputTensor.Data()

	for i := range outputData {
		if outputData[i] < 0 {
			outputData[i] = 0
		}
	}

	outputVariable := &Variable{
		Tensor:       outputTensor,
		Parents:      []*Variable{x},
		RequiresGrad: x.RequiresGrad,
	}

	if x.RequiresGrad {
		outputVariable.BackwardFunc = func() {

			// Does the same to gradient (0 if < 1, else 1)
			grad := outputVariable.Grad.Copy()
			gradData := grad.Data()

			for i := range outputData {
				if outputData[i] < 0 {
					gradData[i] = 0
				}
			}

			if x.Grad == nil {
				x.Grad = grad

			} else {
				x.Grad, _ = x.Grad.Add(grad)
			}
		}
	}

	return outputVariable
}
