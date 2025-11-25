#include <iostream>
#include "tensor.hpp"
#include "layers.hpp"

int main() {
    std::cout << "Project 2: CUDA Mini ConvNet\n";

    Tensor x(1, 1, 8, 8);
    x.random_normal(0.0f, 1.0f);
    x.print_shape("Input");

    Conv2d conv(1, 4, 3);
    ReLU relu;
    MaxPool2x2 pool;

    Tensor y1 = conv.forward(x);
    y1.print_shape("[Conv2D] :");

    Tensor y2 = relu.forward(y1);
    y2.print_shape("[ReLU] :");

    Tensor y3 = pool.forward(y2);
    y3.print_shape("[MaxPool] :");

    std::cout << "Done.\n";
    return 0;
}
