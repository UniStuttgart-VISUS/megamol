#include <iostream>
#ifndef SIZE_T
#define SIZE_T size_t
#endif
#ifndef INT32
#define INT32 int
#endif
#include "mmcore/api/MegaMolCore.h"

int main(int argc, char **argv) {
    unsigned short ver[4];
    ::mmcGetVersionInfo(&ver[0], &ver[1], &ver[2], &ver[3], nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    std::cout << "Using MegaMol Core (V.:" << ver[0] << "." << ver[1] << "." << ver[2] << "." << ver[3] << ")" << std::endl;

    int mod_cnt = ::mmcModuleCount();
    std::cout << "Core backdoor reported " << mod_cnt << " Module descriptions" << std::endl;

    return 0;
}
