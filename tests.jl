
# Pkg.add("Flux")
using Flux
# using Flux:params

# using Zygote
push!(LOAD_PATH, "/Users/mikesaint-antoine/Desktop/new_mikerograd") 
# change this to the location of the folder where mikerograd.jl is on your computer
using mikerograd




########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# Value tests


rand_lower = -20
rand_upper = 20

num1_val = rand_lower + (rand_upper - rand_lower) * rand()
num2_val = rand_lower + (rand_upper - rand_lower) * rand()

num1 = Value(num1_val)
num2 = Value(num2_val)

#############################################################################
# Addition, Value + Value

num3 = num1 + num2

backward(num3)

flux_add(a,b) = a + b

grad = gradient(flux_add, num1_val, num2_val)


if (abs(num1.grad - grad[1]) < 1e-6) && (abs(num2.grad - grad[2]) < 1e-6) && (abs(num3.data - (num1_val + num2_val)) < 1e-6)
    println("Addition, Value + Value: PASS")
else
    println("Addition, Value + Value: FAIL")
    println(num1)
    println(num2)
    println()
    println(num1.grad)
    println(grad[1])
    println()
    println(num2.grad)
    println(grad[2])
    println()

end

#############################################################################
# Subtraction, Value - Value

num1_val = rand_lower + (rand_upper - rand_lower) * rand()
num2_val = rand_lower + (rand_upper - rand_lower) * rand()

num1 = Value(num1_val)
num2 = Value(num2_val)

num3 = num1 - num2

backward(num3)

flux_subtract(a,b) = a - b

grad = gradient(flux_subtract, num1_val, num2_val)


if (abs(num1.grad - grad[1]) < 1e-6) && (abs(num2.grad - grad[2]) < 1e-6) && (abs(num3.data - (num1_val - num2_val)) < 1e-6)
    println("Subtraction, Value - Value: PASS")
else
    println("Subtraction, Value - Value: FAIL")
end

#############################################################################
# Multiplication, Value * Value

num1_val = rand_lower + (rand_upper - rand_lower) * rand()
num2_val = rand_lower + (rand_upper - rand_lower) * rand()

num1 = Value(num1_val)
num2 = Value(num2_val)

num3 = num1 * num2

backward(num3)

flux_multiply(a,b) = a * b

grad = gradient(flux_multiply, num1_val, num2_val)


if (abs(num1.grad - grad[1]) < 1e-6) && (abs(num2.grad - grad[2]) < 1e-6) && (abs(num3.data - (num1_val * num2_val)) < 1e-6)
    println("Multiplication, Value * Value: PASS")
else
    println("Multiplication, Value * Value: FAIL")
end

#############################################################################
# Division, Value / Value

num1_val = rand_lower + (rand_upper - rand_lower) * rand()
num2_val = rand_lower + (rand_upper - rand_lower) * rand()

num1 = Value(num1_val)
num2 = Value(num2_val)

num3 = num1 / num2

backward(num3)

flux_divide(a,b) = a / b

grad = gradient(flux_divide, num1_val, num2_val)


if (abs(num1.grad - grad[1]) < 1e-6) && (abs(num2.grad - grad[2]) < 1e-6) && (abs(num3.data - (num1_val / num2_val)) < 1e-6)
    println("Division, Value / Value: PASS")
else
    println("Division, Value / Value: FAIL")
end


#############################################################################
# e^x, exp(Value)

num1_val = rand_lower + (rand_upper - rand_lower) * rand()
num2_val = rand_lower + (rand_upper - rand_lower) * rand()

num1 = Value(num1_val)

num2 = exp(num1)

backward(num2)

flux_exp(a) = exp(a)

grad = gradient(flux_exp, num1_val)


if (abs(num1.grad - grad[1]) < 1e-6)  && (abs(num2.data - exp(num1_val)) < 1e-6)
    println("e^x, exp(Value): PASS")
else
    println("e^x, exp(Value): FAIL")
end


#############################################################################
# natural log, log(Value)

num1_val = rand_lower + (rand_upper - rand_lower) * rand()
num2_val = rand_lower + (rand_upper - rand_lower) * rand()

num1 = Value(num1_val)

num2 = log(num1)

backward(num2)

flux_log(a) = log(a)

grad = gradient(flux_log, num1_val)


if (abs(num1.grad - grad[1]) < 1e-6)  && (abs(num2.data - log(num1_val)) < 1e-6)
    println("natural log, log(Value): PASS")
else
    println("natural log, log(Value): FAIL")
end





println("done")
