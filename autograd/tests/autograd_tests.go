package tests

import (
	"fmt"

	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/tensor"
)

func TestScalarBackward() {
	fmt.Println("\n\n----- Test Backward Reg #1 -----")
	fmt.Println("Doing y = (x * x) + x")
	fmt.Println("-> dy/dx = 2x + 1 which at x = 3, dy/dx = 7")

	xTensor := tensor.NewTensor([]int{1}, []float32{3.0})
	x := autograd.NewVariable(xTensor, true)
	y := autograd.Add(autograd.Mul(x, x), x)

	autograd.Backward(y) // Computes gradients

	// Should be 7
	fmt.Printf("\nAt x = %.2f, dy/dx = %.2f\n", x.Tensor.Data()[0], x.Grad.Data()[0])
}

func TestMatMulBackward() {
	X := autograd.NewVariable(
		tensor.NewTensor([]int{2, 2}, []float32{
			1, 2,
			3, 4,
		}),
		false,
	)

	W := autograd.NewVariable(
		tensor.NewTensor([]int{2, 1}, []float32{
			0.5,
			-1.0,
		}),
		true,
	)

	Y := autograd.Matrix2dMul(X, W)
	L := autograd.Sum(Y, 0, false)

	autograd.Backward(L)

	fmt.Println("Grad W:", W.Grad)
}

func NumericalGrad(f func() float32, w *tensor.Tensor, eps float32) float32 {
	orig := w.Data()[0]

	w.Data()[0] = orig + eps
	p := f()

	w.Data()[0] = orig - eps
	m := f()

	w.Data()[0] = orig
	return (p - m) / (2 * eps)
}

func TestNumericGrad() {
	W := tensor.NewTensor([]int{1}, []float32{2.0})
	w := autograd.NewVariable(W, true)

	f := func() float32 {
		y := autograd.Mul(w, w)
		return y.Tensor.Data()[0]
	}

	y := autograd.Mul(w, w)
	autograd.Backward(y)

	analytic := w.Grad.Data()[0]
	numeric := NumericalGrad(f, W, 1e-3)

	fmt.Println("Analytic:", analytic)
	fmt.Println("Numeric:", numeric)
}

func TestTrainingLoop() {
	X := tensor.NewTensor([]int{4, 1}, []float32{1, 2, 3, 4})
	Y := tensor.NewTensor([]int{4, 1}, []float32{2, 4, 6, 8})

	w := autograd.NewVariable(
		tensor.NewTensor([]int{1, 1}, []float32{0.1}),
		true,
	)

	b := autograd.NewVariable(tensor.NewScalar(1), true)

	// TODO: Maybe just add scalar support so that I don't need to go through
	//		 a tensor constructor everytime

	learningRate := tensor.NewScalar(0.01)

	// 100 iterations
	for epoch := range 100 {

		// Reset gradients
		w.SetGradToNil()
		b.SetGradToNil()

		// y = weight(x) + bias
		pred := autograd.Add(autograd.Matrix2dMul(
			autograd.NewVariable(X, false),
			w,
		), b)

		// Finds how much prediction differs from true value
		diff := autograd.Sub(pred, autograd.NewVariable(Y, false))

		// MSE (Mean Squared Error)
		loss := autograd.Sum(autograd.Mul(diff, diff), 0, false)
		autograd.Backward(loss)

		// SGD (Stochastic Gradient Descent)
		wUpdate, _ := w.Grad.Mul(learningRate)
		w.Tensor, _ = w.Tensor.Sub(wUpdate)

		bUpdate, _ := b.Grad.Mul(learningRate)
		b.Tensor, _ = b.Tensor.Sub(bUpdate)

		if epoch%10 == 0 {
			/* fmt.Println("\nEpoch", epoch+10)
			fmt.Println("w:", w.Tensor.Data()[0]) */
			fmt.Println("Loss:", loss.Tensor.Data()[0])
		}
	}
}
