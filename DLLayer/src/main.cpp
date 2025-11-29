#include <iostream>
#include <vector>
#include "tensor.hpp"
#include "layers.hpp"

int main() {
    std::cout << "Project 2: CUDA Mini ConvNet (2 conv blocks + FC)\n";

    Tensor x(1, 1, 32, 32);
    x.random_normal(0.0f, 1.0f);
    x.print_shape("Input");

    Conv2d conv1(1, 8, 3);
    ReLU relu1;
    MaxPool2x2 pool1;

    Tensor h1 = conv1.forward(x);
    h1.print_shape("[After Conv1]");

    Tensor h2 = relu1.forward(h1);
    h2.print_shape("[After ReLU1]");

    Tensor h3 = pool1.forward(h2);
    h3.print_shape("[After Pool1]");

    Conv2d conv2(8, 16, 3);
    ReLU relu2;
    MaxPool2x2 pool2;

    Tensor h4 = conv2.forward(h3);
    h4.print_shape("[After Conv2]");

    Tensor h5 = relu2.forward(h4);
    h5.print_shape("[After ReLU2]");

    Tensor h6 = pool2.forward(h5);
    h6.print_shape("[After Pool2]");

    Flatten flatten;
    Tensor flat = flatten.forward(h6);
    flat.print_shape("[After Flatten]");

    int feature_dim = flat.w;

    LinearCPU fc(feature_dim, 10);
    std::vector<float> logits = fc.forward(flat);

    std::cout << "Output logits (size " << logits.size() << "):\n";
    for (size_t i = 0; i < logits.size(); ++i) {
        std::cout << "  class " << i << ": " << logits[i] << "\n";
    }

    std::cout << "Done.\n";
    return 0;
}
