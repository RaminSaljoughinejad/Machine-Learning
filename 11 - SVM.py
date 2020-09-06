"""
    A SVM has a large list of applicable uses.
    However, in machine learning it is typically used for classification.
    It is a powerful tool that is a good choice for classifying complicated
    data with a high degree of dimensions(features).

    Note that K-Nearest Neighbors does not perform well on high-dimensional data.

    n short a support vector machine works by dividing data into multiple classes
    using something called a hyper-plane. A hyper plane is a fancy word for something
    that is straight that can divide data points. In 2D space a hyper-plane is simply
    a line, in 3D space a hyper-plane is a plane. In any space higher than 3D it is simply called a hyper-plane.

    When we create a hyper-plane we need to do the following.
    We must pick two points that are known as our support vectors.

    These points must be the two closest points to the hyper-plane and
    their distance from the hyper-plane must be identical.

     we want farthest hyper plane!
    Kernels
        Kernels provide a way for us to create a hyper-plane for data like seen above.
        We use a kernel to bring our data up to a higher dimension (in this case from 2D->3D).
        We hope that by doing this we will have our points plotted in a way that we can divide them using a hyper-plane.

        By applying a kernel to our data above we hope to get something that looks like ...
        Kernels
            - Linear
            – Polynomial
            – Circular
            – Hyperbolic Tangent (Sigmoid)

    Soft margin
        allowing hyper prameter
    Hard margin
        not allowing prameters

"""