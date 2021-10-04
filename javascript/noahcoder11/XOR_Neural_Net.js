// Matrix Operations
function randomArr(w, h){
  var arr = []
  
  for(var i = 0;i < h;i++){
    arr[i] = []
    
    for(var j = 0;j < w;j++){
      arr[i][j] = Math.random()
      
    }
  }
  
  return arr
}

function dotProduct(m1, m2){
  var dot = []
  
  for(var i = 0;i < m1.length;i++){
    dot[i] = [];
    
    for(var j = 0;j < m2[0].length;j++){
      dot[i][j] = 0
      
      for (var k = 0; k < m1[0].length;k++){
        dot[j] = m1[i][k] * m2[k][j]
        
      }
    }
  }
  
  return dot
}

function add(m1, m2){
  var a = []
  
  if(m2.length === 1){
    for(var i = 0;i < m1.length;i++){
      a[i] = []
      
      for(var j = 0;j < m1[0].length;j++){
        a[i][j] = m1[i][j] + m2[0][j]
        
      }
    }
  }else {
    for(var i = 0;i < m1.length;i++){
      a[i] = []
      
      for(var j = 0;j < m1[0].length;j++){
        a[i][j] = m1[i][j] + m2[i][j]
        
      }
    }
  }
  
  return a
}
function subtract(m1, m2){
  var s = []
  
  if(m2.length === 1){
    for(var i = 0;i < m1.length;i++){
      s[i] = []
      
      for(var j = 0;j < m1[0].length;j++){
        s[i][j] = m1[i][j] + m2[0][j]
        
      }
    }
  }else {
    for(var i = 0;i < m1.length;i++){
      s[i] = []
      
      for(var j = 0;j < m1[0].length;j++){
        s[i][j] = m1[i][j] + m2[i][j]
        
      }
    }
  }
  
  return s
}

function transpose(m, config){
  var arr = []
  
  for(var i = 0;i < m[0].length;i++){
    arr[i] = []
    
  }
  for(var i = 0;i < m.length;i++){
    for(var j = 0;j < m[0].length;j++){
      arr[j][i] = m[i][j]
      
    }
  }
  
  return arr
}
function subtract(m1, m2){
  var s = []
  
  if(m2.length === 1){
    for(var i = 0;i < m1.length;i++){
      s[i] = []
      
      for(var j = 0;j < m1[0].length;j++){
        s[i][j] = m1[i][j] + m2[0][j]
        
      }
    }
  }else {
    for(var i = 0;i < m1.length;i++){
      s[i] = []
      
      for(var j = 0;j < m1[0].length;j++){
        s[i][j] = m1[i][j] + m2[i][j]
        
      }
    }
  }
  
  return s
}

function scale(m, scalar){
  var arr = []
  
  for(var i = 0;i < m.length;i++){
    arr[i] = []
    for(var j = 0;j < m[0].length;j++){
      arr[i][j] = m[i][j] * scalar
      
    }
  }
  
  return arr
}
function applyFunc(m, f){
  var arr = []
  
  for(var i = 0;i < m.length;i++){
    arr[i] = []
    for(var j = 0;j < m[0].length;j++){
      arr[i][j] = f(m[i][j])
      
    }
  }
  
  return arr
}

var relu = {
  forward: function(x){
    return Math.max(0, x)
  },
  backward: function(x){
    return x > 0 ? 1 : 0
  }
}
var sigmoid = {
  forward: function(x){
    return 1 / (1 + Math.exp(-x))
  },
  backward: function(x){
    return this.forward(x) * (1 - this.forward(x))
  }
}

class Activation {
  constructor(a){
    this.a = a
  }
  forward(input){
    return applyFunc(input, this.a.forward)
  }
  backward(input){
    return applyFunc(input, this.a.backward)
  }
}

// Dense Layer
class Dense {
  constructor(inputs, neurons, activation){
    this.weights = randomArr(neurons, inputs)
    this.bias = randomArr(neurons, 1)
    this.activation = activation
  }
  forward(input){
    this.input = input
    
    this.output = add(dotProduct(this.input, this.weights), this.bias)
    
    return this.activation.forward(this.output)
  }
  backward(output, learningRate){
    output = this.activation.backward(output)
    var weightGradient = dotProduct(transpose(this.input), output)
    var inputGradient = dotProduct(output, transpose(this.weights))
    
    this.weights = subtract(this.weights, scale(weightGradient, learningRate))
    this.bias = subtract(this.bias, scale(output, learningRate))
    
    return inputGradient
  }
}

// Mean Squared Error
class Loss {
  constructor(){}
  forward(predictions, target){
    var loss = 0
    var s = subtract(target, predictions)
    
    for(var i = 0;i < s.length;i++){
      for(var j = 0;j < s[0].length;j++){
        loss += Math.pow(s[i][j], 2)
      }
    }
    
    loss /= target.length
    
    return loss
  }
  backward(predictions, target){
    return scale(subtract(this.prediction, this.target), 2/predictions.length)
  }
}

// Neural Network
class NeuralNetwork {
  constructor(layers, loss){
    this.loss = loss
    this.layers = layers
    this.bestLoss = 10e9
  }
  forward(input){
    var output = input
    
    for(var i = 0;i < this.layers.length;i++){
      output = this.layers[i].forward(output)
    }
    
    return output
  }
  backward(ouput, learningRate){
    var input = output
    var reversed = this.layers.slice().reverse()
    
    for(var i = 0;i < reversed.length;i++){
      input = reversed[i].backward(input, learningRate)
    }
    
    return input
  }
  fit(xTrain, yTrain, epochs, learningRate){
    var history = {
      loss:[]
    }
    
    for(var e = 0;e < epochs;e++){
      var p = this.forward(xTrain)
      var l = this.loss.forward(p, yTrain)
      
      history.loss.push(l)
      
      var lP = this.loss.backward(p, yTrain)
      this.backward(lP, learningRate)
    }
  }
}

var X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]
var Y = [
  [0],
  [1],
  [1],
  [0],
]

var net = new NeuralNetwork(
  [
    new Dense(2, 2, new Activation(relu)),
    new Dense(2, 4, new Activation(relu)),
    new Dense(4, 1, new Activation(sigmoid)),
  ],
  new Loss()
)

net.fit(X, Y, 100, 0.01)
var predictions = net.forward(X)
