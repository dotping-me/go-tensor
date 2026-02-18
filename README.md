# Go CPU-First Deep Learning Framework
A deep learning framework written in Go meant to provide ML tooling to the Go ecosystem to bridge the gap between Python and Go in the AI space.

## ⚙️ How it works?
At the core sits a `Tensor` and `Autograd` engine, implemented from scratch.

1. The Tensor Engine 📦
    - Basically implements Tensor operations such as *Broadcasting*, *Batched Matrix Multiplication* and so on.
    - Supports *N-D* tensors.
    - Currently supports only `float32` data!

2. The Autograd Engine 📈
    - Keeps track of gradients and implements backwards propagation.
    - It is built upon the Tensor Engine, using its underlying implementations.  
    The `Model Definition API` (What the user will use to define models, training, etc...) is **built upon the Autograd Engine!**
    - It provides implementation for *Gradient Tracking*, *Activation Functions*, *Loss Functions* and so on.

## 📚 The Model Definition API
This is what the user will use to define models, train, run inference and everything else!

### Model
A wrapper for all the AI/ML logic inside!

| F(x) | Args | Returns | Desc |
| ---- | ---- | ------- | ---- |
| `nn.NewModel(root)` | Takes in a pointer to a base root layer/topology such as `topologies.Sequential` | `*Model` | Creates a model |
| I'll continue writing this later... | &nbsp; | &nbsp; | &nbsp; |

### Topologies
Defining layers and how data flows!

| F(x) | Args | Returns | Desc |
| ---- | ---- | ------- | ---- |
| `topologies.NewModel()` | `None` | `*topologies.Sequential` | Creates a Sequential Layer container |
| `seq.Add(layer)` | Takes in a pointer to a layer such as `layers.DenseLayer` | `None` | Adds a layer to the topology |
| I'll continue writing this later... | &nbsp; | &nbsp; | &nbsp; |

***NOTE:*** *Currently there is support for only `Sequential` topology!*

### Layers
***TODO:*** *Write docs!*

### Loss Functions
***TODO:*** *Write docs!*

### Optimizers
***TODO:*** *Write docs!*

### Storage and HTTP Serving
***TODO:*** *Write docs!*

### Autograd Utilities
***TODO:*** *Write docs!*

## 🧪 Example: Real Estate Pricing Prediction
See `/tests/test_real_estate_example.go` for the full implementation!

### Defining and Fitting the Model!
Training the model to identify patterns in Real Estate pricing!

```go
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

seq := topologies.NewSequential()
seq.Add(layers.NewDenseLayer(2, 1)) // 2 Inputs -> 1 Output

model := nn.NewModel(seq)
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
}
```

### Running Inference
Testing the fitted model with adjusted parameters!

```go
// ---------------------------------
//   Testing the model (Inference)
// ---------------------------------

testInput := autograd.NewVariable(
    tensor.NewTensor([]int{1, 2}, []float32{150, 3}), false)

// Inference -> y = Wx + b
testOutput := model.Predict(testInput)
fmt.Println(testOutput.Tensor.Data()[0]) // Outputs prediction
```

```bash
107458.476562
```

<!--
## 📂 Project Directory Structure
This is how the current project is structured!

```bash
/
├── autograd/
├── nn
│   ├── layers/
│   ├── losses/
│   ├── optimizers/
│   ├── storage/
│   ├── topologies/
│   ├── utils/
│   │
│   └── nn.go
│
├── tensor/
├── tests/
│
├── README.md
├── .gitignore
├── go.mod
└── main.go
``` 
-->

## 🎯 Tasklist
These are the things that are currently being worked on!

1. NLP and Transformer Implementation
    - Currently fixing bugs for it.
    - See `tests/test_transformer.go` for the test case!

2. Maybe merging the `Tensor` and `Autograd` into one singular backbone? (i.e. `backbone/`)

3. Review, test and sanitize Tensor and Autograd operations, especially operations involving shape transformation!

4. Write proper documentation!