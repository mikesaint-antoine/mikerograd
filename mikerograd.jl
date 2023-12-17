module mikerograd
export Value,show, +, *, exp, ^, inv, / , tanh, -, backward, Tensor, relu, softmax_crossentropy


using Random
using Statistics



## Hi, thanks for checking out mikerograd! 

## I started this project as a learning exercise for myself, since I'm a beginner at Julia and machine learning.
## Figured other people might find it useful too, so I'm putting it up on my Github.

## Basically the goal here is to make some basic gradient-tracking objects, with code that is
## simple enough to read, understand, and edit.

## There are 2 classes here.

## Value 
##  - basically a Julia version of the Value class in Andrej Karpathy's original Python Micrograd.
##  - scalar-valued gradient tracking
##  - great tool for learning the basics of backpropogation
##  - possibly useful for simple examples like fitting a line to data by gradient descent.

## Tensor 
##  - something new that I decided to add myself
##  - tensor-valued gradient tracking (arrays and matrices, rather than single numbers)
##  - faster than the Value class, and a bit more similar to how real ML libraries actually work.
##  - still simple enough to read and understand the code
##  - useful for some basic neural net applications, like MNIST

## Please see the demo.ipynb notebook for an introduction on how to use these classes.

## Anyway, I hope some people will find this helpful as a learning tool for Julia and machine learning!
## If you have any questions or run into problems with the code, please feel free to email me at mikest@udel.edu.


## Citations (and great sources for learning the basics of neural nets in Python):
##  - Andrej Karpathy's Micrograd video lesson: https://www.youtube.com/watch?v=VMj-3S1tku0 
##  - Harrison Kinsley (Sentdex on Youtube) neural net textbook: https://nnfs.io/ 

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
##
## Value 
##
## this is a Julia version of Andrej Karpathy's original Python Micrograd:
## https://github.com/karpathy/micrograd
##
## great video tutorial by Karpathy here:
## https://www.youtube.com/watch?v=VMj-3S1tku0
##
##
## basically a scalar-valued gradient-tracking object
##
## stores numbers, lets you do operations with them, and automatically tracks the 
## derivative of the output with respect to all the inputs.

###################################
## operations currently supported:

## addition
## +(a::Value, b::Value)
## +(a::Value, b::Number)
## +(a::Number, b::Value)

## multiplication
## *(a::Value, b::Value)
## *(a::Value, b::Number)
## *(a::Number, b::Value)

## division
## /(a::Value, b::Value) for now, both must be Values. might make more robust in the future


## negation and subtraction
## -(a::Value)
## -(a::Value, b::Value)
## -(a::Value, b::Number)
## -(a::Number, b::Value)


## exp(a::Value)
## log(a::Value)

## exponent (must be number, not Value) 
## ^(a::Value, b::Number)

## x^(-1) -- Julia requires a special case for this
## inv(a::Value)

## tanh(a::Value)

## full backward pass to calculate gradients: backward(a::Value)


#################################################################

## basic type definition and constructor
mutable struct Value
    data::Float64
    grad::Float64
    _children::Array{Value, 1} # keeps track of the operands used in an operation
    _backward::Function

    function Value(data::Number; children::Array{Value, 1} = Value[])
        value = new(Float64(data), 0.0, children) # 0.0 here is grad initialization
        value._backward = () -> nothing  # default backward function that does nothing
        return value
    end
end
        

## defines what happens when we print out a Value
import Base: show
function show(io::IO, value::Value)
    print(io, "Value(",value.data, ")")
end



#################################################################
## addition 

# defines addition for 2 Value objects
import Base: +
function +(a::Value, b::Value)
    out = a.data + b.data

    result = Value(out, children=[a, b])

    function _backward()
        a.grad += 1.0 * result.grad
        b.grad += 1.0 * result.grad
    end
    result._backward = _backward


    return result
end

# addition for Value + number
function +(a::Value, b::Number)
    b_value = Value(Float64(b)) # cast b to Value
    return a + b_value  # use the existing method for Value + Value
end

# addition for number + Value
function +(a::Number, b::Value)
    return b + a # use Value + Number, which then casts the number to Value and does Value + Value
end



#################################################################
## multiplication

# multiplication for 2 Value objects
import Base: *
function *(a::Value, b::Value)
    out = a.data * b.data

    result = Value(out, children=[a, b])

    function _backward()
        a.grad += b.data * result.grad
        b.grad += a.data * result.grad
    end
    result._backward = _backward

    return result
end


# Value * number
function *(a::Value, b::Number)
    b_value = Value(Float64(b))  # cast b to Value
    return a * b_value # use the existing method for Value * Value
end

# number * Value
function *(a::Number, b::Value)
    return b * a # use Value * Number, which then casts the number to Value and does Value * Value
end


#################################################################

# e^x
import Base: exp
function exp(a::Value)
    x = a.data
    out_val = exp(x)

    result = Value(out_val, children=[a])

    function _backward()
        a.grad += out_val * result.grad
    end
    result._backward = _backward

    return result
end


import Base: log
function log(a::Value)
    x = a.data
    out = log(x)

    result = Value(out, children=[a])

    function _backward()
        a.grad += (1.0 / x) * result.grad
    end
    result._backward = _backward

    return result
end

#################################################################

# x^c
import Base: ^
function ^(a::Value, b::Number)

    # IMPORTANT -- exponent must be an int or float, not currently supporting Value object exponent
    # might change this in the future

    out = a.data ^ b

    result = Value(out, children=[a])

    function _backward()
        a.grad += b * (a.data ^ (b - 1)) * result.grad
    end
    result._backward = _backward

    return result
end

# need special case for x^(-1) in Julia
import Base: inv
function inv(a::Value)

    out = 1.0 / a.data

    result = Value(out, children=[a])

    function _backward()
        a.grad -= (1.0 / (a.data * a.data)) * result.grad
    end
    result._backward = _backward

    return result
end


#################################################################

# division: a / b = a * b^(-1)
import Base: /
function /(a::Value, b::Value)
    return a * (b ^ -1)
end

# for now both must be Values
# TODO -- implement Value / number and number / Value at some point? 


#################################################################
# negation and subtraction

import Base: -

# negation
function -(a::Value)
    return a * -1
end

# subtraction: Value - Value
function -(a::Value, b::Value)
    return a + (-b)
end

# Value - number
function -(a::Value, b::Number)
    b_value = Value(Float64(b))
    return a - b_value
end


# number - Value
function -(a::Number, b::Value)
    return b - a
end


#################################################################


# tanh
import Base: tanh
function tanh(a::Value)
    x = a.data
    out = (exp(2 * x) - 1) / (exp(2 * x) + 1)

    result = Value(out, children=[a])

    function _backward()
        a.grad += (1 - out^2) * result.grad
    end
    result._backward = _backward

    return result
end


# full backward pass
function backward(a::Value)

    function build_topo(v::Value, visited=Value[], topo=Value[])
        if !(v in visited)
            push!(visited, v)
            for child in v._children
                build_topo(child, visited, topo)
            end
            push!(topo, v)
        end
        return topo
    end
    
    topo = build_topo(a)

    a.grad = 1.0
    for node in reverse(topo)
        node._backward()
    end

end

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
##
## Tensor 
##
## unfortunately the Value class is too slow to do even simple neural net examples like MNIST
## so I decided to add a Tensor class, which is faster and more similar to how real neural net libraries work
##
## the Tensors is similar to the Values, except they store arrays and matrices rather than single numbers
## right now, they only have the bare minimum operations needed to solve MNIST
## just the basic forward and backward passes of a regular neural net with some dense layers
## might add more functionality later


###################################
## operations currently supported:

## addition
## +(a::Tensor, b::Tensor)
## can be used to add biases if one input is 2D and the other is 1D

## matrix-multiplication / dot product
## *(a::Tensor, b::Tensor)

## relu activation function
## relu(a::Tensor)

## softmax activation / cross entropy loss combination
## softmax_crossentropy(a::Tensor,y_true::Union{Array{Int,2},Array{Float64,2}}

## IMPORTANT -- y_true must be one-hot encoded 2D matrix -- just regular matrix, not Tensor
## might change this in the future

## full backward pass to calculate gradients: backward(a::Tensor)


mutable struct Tensor
    data::Union{Array{Float64,1},Array{Float64,2}}
    grad::Union{Array{Float64,1},Array{Float64,2}}
    _children::Array{Tensor, 1}
    _backward::Function

    function Tensor(data::Union{Array{Float64,1},Array{Float64,2}}; children::Array{Tensor, 1} = Tensor[])
        tensor = new(data, zeros(Float64, length(data)) , children) # 0.0 here is grad initialization
        tensor._backward = () -> nothing  # default backward function that does nothing
        return tensor
    end
end


## define what happens when we print out a Tensor
import Base: show
function show(io::IO, tensor::Tensor)
    print(io, "Tensor(",tensor.data, ")")
end




# define addition for 2 Tensor objects
import Base: +
function +(a::Tensor, b::Tensor)

    if length(size(a.data)) == length(size(b.data))
        out = a.data .+ b.data
    elseif length(size(a.data)) > length(size(b.data))
        # a is 2D, b is 1D
        out = a.data .+ transpose(b.data)
    else
        # a is 1D, b is 2D
        out = b.data .+ transpose(a.data)
    end

    result = Tensor(out, children=[a, b])

    function _backward()
        

        if length(size(result.grad)) > length(size(a.data))
            a.grad = dropdims(sum(result.grad, dims=1), dims=1) # need dropdims to make this size (x,) rather than size (1,x)
        else
            a.grad = ones(size(a.data)) .* result.grad
        end

        if length(size(result.grad)) > length(size(b.data))
            b.grad = dropdims(sum(result.grad, dims=1), dims=1) # need dropdims to make this size (x,) rather than size (1,x)
        else
            b.grad = ones(size(b.data)) .* result.grad
        end


    end

    result._backward = _backward

    return result
end





import Base: *
function *(a::Tensor, b::Tensor)
    out = a.data * b.data

    result = Tensor(out, children=[a, b])

    function _backward()
        a.grad = result.grad * transpose(b.data)
        b.grad = transpose(a.data) * result.grad 
    end
    result._backward = _backward

    return result
end



function relu(a::Tensor)

    result = Tensor(max.(a.data), children=[a])

    function _backward()
        a.grad = (a.data .> 0) .* result.grad
    end
    result._backward = _backward

    return result
end





function softmax_crossentropy(a::Tensor,y_true::Union{Array{Int,2},Array{Float64,2}})

    ## implementing softmax activation and cross entropy loss separately leads to very complicated gradients
    ## but combining them makes the gradient a lot easier to deal with

    ## credit to Sendex and his textbook for teaching me this part
    ## great textbook for doing this stuff in Python, you can get it here:
    ## https://nnfs.io/

    # softmax activation
    exp_values = exp.(a.data .- maximum(a.data, dims=2))
    probs = exp_values ./ sum(exp_values, dims=2)
    
    ## crossentropy - sample losses
    samples = size(probs, 1)
    probs_clipped = clamp.(probs, 1e-7, 1 - 1e-7)
    # deal with 0s


    # basically just returns an array with the probability of the correct answer for each batch
    correct_confidences = sum(probs_clipped .* y_true, dims=2)

    # negative log likelihood
    sample_losses = -log.(correct_confidences)


    # loss_mean
    loss_mean = mean(sample_losses)


    result = Tensor([loss_mean], children=[a])

    function _backward()


        samples = size(probs, 1)

        # convert from one-hot to index list
        y_true_argmax = argmax(y_true, dims=2)

        a.grad = copy(probs)
        for samp_ind in 1:samples
            a.grad[samp_ind, y_true_argmax[samp_ind][2]] -= 1
            ## this syntax y_true_argmax[i][2] is just to get the column index of the true value
        end
        a.grad ./= samples

    end
    result._backward = _backward

    return result
end



# full backward pass
function backward(a::Tensor)

    function build_topo(v::Tensor, visited=Tensor[], topo=Tensor[])
        if !(v in visited)
            push!(visited, v)
            for child in v._children
                build_topo(child, visited, topo)
            end
            push!(topo, v)
        end
        return topo
    end
    
    topo = build_topo(a)

    a.grad .= 1.0
    for node in reverse(topo)
        node._backward()
    end
end











end