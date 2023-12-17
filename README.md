# mikerograd

Hi, thanks for checking out **mikerograd**! This is a Julia version of Andrej Karpathy's original [micrograd](https://github.com/karpathy/micrograd) Python library, with an additional Tensor object-type added.

I started this project as a learning exercise for myself, since I'm a beginner at Julia and machine learning. Figured other people might find it useful too, so I'm putting it up on my Github. 

Basically the goal here is to make some basic gradient-tracking objects, with code that's simple enough to read, understand, and edit.

The file **mikerograd.jl** contains 2 object-types (basically classes):

**Value** 
- basically a Julia version of the Value class in Andrej Karpathy's original Python Micrograd.
- scalar-valued gradient tracking
- great tool for learning the basics of backpropogation
- possibly useful for simple examples like fitting a line to data by gradient descent.

**Tensor** 
- something new that I decided to add myself
- tensor-valued gradient tracking (arrays and matrices, rather than single numbers)
- faster than the Value class, and a bit more similar to how real ML libraries actually work.
- still simple enough to read and understand the code
- useful for some basic neural net applications, like MNIST

Please see the **demo.ipynb** notebook for an introduction on how to use these objects. It includes a basic introduction, as well as line-fitting and MNIST examples.

Anyway, I hope some people will find this helpful as a learning tool for Julia and machine learning! If you have any questions or run into problems with the code, please feel free to email me at **mikest@udel.edu**.

**Citations (and great sources for learning the basics of neural nets in Python)**:
- [Andrej Karpathy's Micrograd video lesson](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Harrison Kinsley (Sentdex on Youtube) neural net textbook](https://nnfs.io/)

