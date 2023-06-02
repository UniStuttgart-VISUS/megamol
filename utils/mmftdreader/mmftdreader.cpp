#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct ColumnInfo {
    uint16_t nameLength;
    std::string name;
    uint8_t type;
    float min;
    float max;
};

int main(int argc, char* argv[]) {
    using namespace std::string_literals;

    if (argc != 2) {
        std::cerr << "mmftdreader" << std::endl << "Usage: ./mmftdreader <mmftd file>" << std::endl;
        return 1;
    }

    std::filesystem::path filename(argv[1]);

    if (!std::filesystem::is_regular_file(filename)) {
        std::cerr << "File not found: " << filename.string() << std::endl;
        return 1;
    }

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::cout << "# General Info" << std::endl
              << std::endl
              << "Filename:  " << filename.string() << std::endl
              << "Size:      " << size << std::endl;

    // TODO do complete file validation, now just check magic number
    std::string magic(6, '\0');
    file.read(magic.data(), 6);
    if (magic != "MMFTD\0"s) {
        std::cerr << "Invalid file magic number." << std::endl;
        return 1;
    }

    uint16_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint16_t));
    std::cout << "Version:   " << version << std::endl;

    uint32_t colCount;
    file.read(reinterpret_cast<char*>(&colCount), sizeof(uint32_t));
    std::cout << "Columns:   " << colCount << std::endl;

    std::vector<ColumnInfo> info(colCount);
    for (uint32_t i = 0; i < colCount; i++) {
        file.read(reinterpret_cast<char*>(&info[i].nameLength), sizeof(uint16_t));
        std::vector<char> nameBuf(info[i].nameLength);
        file.read(nameBuf.data(), info[i].nameLength);
        info[i].name = std::string(nameBuf.data());
        file.read(reinterpret_cast<char*>(&info[i].type), sizeof(uint8_t));
        file.read(reinterpret_cast<char*>(&info[i].min), sizeof(float));
        file.read(reinterpret_cast<char*>(&info[i].max), sizeof(float));
    }

    uint64_t rowCount;
    file.read(reinterpret_cast<char*>(&rowCount), sizeof(uint64_t));
    std::cout << "Rows:      " << rowCount << std::endl;

    std::cout << std::endl
              << "# Table Column Info" << std::endl
              << std::endl
              << "| Name                 | Type | Min          | Max          |" << std::endl
              << "|----------------------|------|--------------|--------------|" << std::endl;
    for (const auto& i : info) {
        // clang-format off
        std::cout << "| " << std::left << std::setw(20) << i.name
                  << " | " << std::right << std::setw(4) << (int)i.type
                  << " | " << std::setw(12) << i.min
                  << " | " << std::setw(12) << i.max
                  << " |" << std::endl;
        // clang-format on
    }
    std::cout << "|----------------------|------|--------------|--------------|" << std::endl;

    std::vector<float> data(rowCount * colCount);
    file.read(reinterpret_cast<char*>(data.data()), rowCount * colCount * sizeof(float));

    // clang-format off
    std::cout << std::endl
              << "# Table Data" << std::endl
              << std::endl
              << "| " << std::left;
    // clang-format on

    for (const auto& i : info) {
        std::cout << std::setw(12) << i.name << " | ";
    }
    std::cout << std::endl << "|" << std::right;
    for (const auto& i : info) {
        std::cout << "--------------|";
    }
    std::cout << std::endl;
    for (uint64_t r = 0; r < rowCount; r++) {
        std::cout << "| ";
        for (uint32_t c = 0; c < colCount; c++) {
            std::cout << std::setw(12) << data[r * colCount + c] << " | ";
        }
        std::cout << std::endl;
    }

    return 0;
}
