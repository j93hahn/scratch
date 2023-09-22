#include <iostream>
#include <memory>
#include <torch/torch.h>


int main(int argc, char* argv[]) {
    std::cout << "Begin training!" << std::endl;
    torch::manual_seed(69250713);   // MNJ's random seed

    return 0;
}
