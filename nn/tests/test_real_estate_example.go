package tests

import (
	"fmt"

	"github.com/dotping-me/go-tensor/autograd"
	"github.com/dotping-me/go-tensor/nn"
	"github.com/dotping-me/go-tensor/nn/layers"
	"github.com/dotping-me/go-tensor/nn/losses"
	"github.com/dotping-me/go-tensor/nn/optimizers"
	"github.com/dotping-me/go-tensor/tensor"
)

// Trains a neural network to understand and estime how real estate prices change
func TestRealEstatePrediction() {

	// ------------------
	//   Sample Dataset
	// ------------------

	// X maps the relationship between the size of a house and its # of bedrooms
	X := tensor.NewTensor([]int{4, 2}, []float32{
		50, 1,
		80, 2,
		120, 3,
		200, 4,
	})

	// Y is the price of the real estate
	Y := tensor.NewTensor([]int{4, 1}, []float32{
		35000,
		60000,
		90000,
		140000,
	})

	varX := autograd.NewVariable(X, false)
	varY := autograd.NewVariable(Y, false)

	// -------------------------
	//   Defining the NN model
	// -------------------------

	model := nn.NewSequential()
	model.Add(layers.NewDenseLayer(2, 1)) // 2 Inputs -> 1 Output
	opt := optimizers.NewSGD(model.Parameters(), 0.000001)

	// ----------------------
	//   Training the model
	// ----------------------

	for epoch := range 1000 {
		opt.SetAllParamsGradToNil()
		prediction := model.Forward(varX)

		mse := losses.MeanSquaredError(prediction, varY) // Calculates loss
		autograd.Backward(mse)

		opt.Step() // Updates parameters

		if epoch%10 == 0 {
			fmt.Println(
				"Epoch:", epoch,
				"Loss:", mse.Tensor.Data()[0],
			)
		}
	}

	// -----------------------
	//   Check final weights
	// -----------------------
	fmt.Println("\nParams after training:")
	for _, p := range model.Parameters() {
		fmt.Println(p.Tensor)
	}

	// ---------------------------------
	//   Testing the model (Inference)
	// ---------------------------------

	p := model.Parameters()
	W := p[0].Tensor // Extracting trained weights and biases
	b := p[1].Tensor

	testInput := tensor.NewTensor([]int{1, 2}, []float32{150, 3})

	// TODO: Turn this into a method -> Model.Predict()
	// Inference -> y = Wx + b
	testOutput, _ := testInput.Matrix2DMul(W)
	testOutput, _ = testOutput.Add(b)
	fmt.Printf("\nTesting model with\n%v\n=\n%v\n", testInput, testOutput)
}
