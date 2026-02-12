// Extending tensor operations, adding a callback function to undo their effects
// during backwards pass

package autograd

import (
	"log"

	"github.com/dotping-me/go-tensor/tensor"
)

// TODO: Maybe these can be made into a modular fashion like Tensor elementwise ops
// TODO: Cleanup this file
// TODO: Maybe turn them into Autograd Variable methods???

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

func Div(a, b *Variable) *Variable {
	outputTensor, err := a.Tensor.Div(b.Tensor)
	if err != nil {
		log.Fatal(err)
	}

	outputVariable := &Variable{
		Tensor:       outputTensor,
		Parents:      []*Variable{a, b},
		RequiresGrad: a.RequiresGrad || b.RequiresGrad,
	}

	// Formula:
	// C = A / B

	// Therefore:
	// gC/gA = grad / b (How does that even make sense?)
	// gC/gB = grad * (a / (b**2) )

	if outputVariable.RequiresGrad {
		outputVariable.BackwardFunc = func() {

			if a.RequiresGrad {
				gradA, _ := outputVariable.Grad.Div(b.Tensor)
				SumGrad(&a.Grad, gradA)
			}

			if b.RequiresGrad {
				bSquared, _ := b.Tensor.Mul(b.Tensor)
				gradB, _ := a.Tensor.Div(bSquared) // (a / (b**2) )

				gradB, _ = outputVariable.Grad.Mul(gradB)
				gradB, _ = gradB.Neg()
				SumGrad(&b.Grad, gradB)
			}
		}
	}

	return outputVariable
}

func Pow(a *Variable, factor float32) *Variable {
	outputTensor, _ := a.Tensor.Pow(factor)

	outputVariable := &Variable{
		Tensor:       outputTensor,
		Parents:      []*Variable{a},
		RequiresGrad: a.RequiresGrad,
	}

	// Formula:
	// y = x^n

	// Therefore:
	// dy/dx = n * x^(n-1)

	if outputVariable.RequiresGrad {
		outputVariable.BackwardFunc = func() {
			powMinus1, _ := a.Tensor.Pow(factor - 1)
			scale, _ := powMinus1.Mul(tensor.NewScalar(factor))
			grad, _ := outputVariable.Grad.Mul(scale)
			SumGrad(&a.Grad, grad)
		}
	}

	return outputVariable
}

func Sqrt(a *Variable) *Variable {
	outputTensor, _ := a.Tensor.Sqrt()

	outputVariable := &Variable{
		Tensor:       outputTensor,
		Parents:      []*Variable{a},
		RequiresGrad: a.RequiresGrad,
	}

	// Basicaaaallyyyyyy
	// y = x^0.5
	// dy/dx = 0.5 * x^-0.5

	if outputVariable.RequiresGrad {
		outputVariable.BackwardFunc = func() {
			denom, _ := outputTensor.Mul(tensor.NewScalar(2))
			grad, _ := outputVariable.Grad.Div(denom)
			SumGrad(&a.Grad, grad)
		}
	}

	return outputVariable
}

func Neg(a *Variable) *Variable {
	outputTensor, _ := a.Tensor.Neg()

	outputVariable := &Variable{
		Tensor:       outputTensor,
		Parents:      []*Variable{a},
		RequiresGrad: a.RequiresGrad,
	}

	// I don't know for sure about this one

	if outputVariable.RequiresGrad {
		outputVariable.BackwardFunc = func() {
			negGrad, _ := outputVariable.Grad.Neg()
			SumGrad(&a.Grad, negGrad)
		}
	}

	return outputVariable
}

func Log(a *Variable) *Variable {
	outputTensor, _ := a.Tensor.Log()

	outputVariable := &Variable{
		Tensor:       outputTensor,
		Parents:      []*Variable{a},
		RequiresGrad: a.RequiresGrad,
	}

	// Formula:
	// y = log(x)
	// dy/dx = 1/x

	if outputVariable.RequiresGrad {
		outputVariable.BackwardFunc = func() {
			inv, _ := a.Tensor.Pow(-1)
			grad, _ := outputVariable.Grad.Mul(inv)
			SumGrad(&a.Grad, grad)
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

func Mean(a *Variable, axisIndex int, keepAxis bool) *Variable {
	// NOTE: Gradient accumulation is handled through other autograd op functions

	// Axis Normalization
	if axisIndex < 0 {
		axisIndex = len(a.Tensor.Shape()) + axisIndex
	}

	sum := Sum(a, axisIndex, keepAxis)
	size := float32(a.Tensor.Shape()[axisIndex])
	scaleTensor := tensor.NewScalar(1.0 / size)

	scaleVariable := NewVariable(scaleTensor, false)
	return Mul(sum, scaleVariable)
}

// I forgot the formula but it should be there somewhere
func Variance(a *Variable, axisIndex int, keepAxis bool) *Variable {
	mean := Mean(a, axisIndex, true)

	diff := Sub(a, mean)
	sq := Mul(diff, diff)
	variance := Mean(sq, axisIndex, keepAxis)

	return variance
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
