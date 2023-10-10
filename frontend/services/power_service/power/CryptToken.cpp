#include "CryptToken.h"

#include <filesystem>
#include <fstream>

#if WIN32
#include <Windows.h>
#include <wincrypt.h>
#if MEGAMOL_USE_OPENGL
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WGL
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif
#endif

namespace megamol::power {
CryptToken::CryptToken(std::string const& filename, void* window_ptr = nullptr) : token_safe_(nullptr), token_size_(0) {
    std::filesystem::path filepath(filename);
    if (std::filesystem::exists(filepath) && std::filesystem::is_regular_file(filepath)) {
        auto in_file = std::ifstream(filepath, std::ios::binary);
        token_size_ = std::filesystem::file_size(filepath);
        auto data_size = token_size_;
        auto cbMod = data_size % CRYPTPROTECTMEMORY_BLOCK_SIZE;
        if (cbMod) {
            data_size += CRYPTPROTECTMEMORY_BLOCK_SIZE - cbMod;
        }
        token_safe_ = (char*)LocalAlloc(LPTR, data_size);
        if (!token_safe_) {
            throw std::runtime_error("[CryptToken] Cannot allocate data");
        }
        if (!CryptProtectMemory(token_safe_, data_size, CRYPTPROTECTMEMORY_SAME_PROCESS)) {
            LocalFree(token_safe_);
            throw std::runtime_error("[CryptToken] Cannot encrypt data");
        }
        std::string file_data;
        file_data.resize(token_size_);
        in_file.read(file_data.data(), token_size_);
        DATA_BLOB blob_in;
        blob_in.cbData = token_size_;
        blob_in.pbData = (BYTE*)(file_data.data());
        DATA_BLOB blob_out;
        blob_out.cbData = data_size;
        blob_out.pbData = (BYTE*)token_safe_;
        if (!CryptUnprotectData(&blob_in, nullptr, nullptr, nullptr, nullptr, 0, &blob_out)) {
            LocalFree(token_safe_);
            throw std::runtime_error("[CryptToken] Cannot copy data");
        }
        token_size_ = data_size;
        in_file.close();
    } else {
#if MEGAMOL_USE_OPENGL
        char* buffer = new char[32767];
        GetEnvironmentVariable("DataverseAPIKey", buffer, 32767);
#else
        throw std::runtime_error("[CryptToken] Cannot open file");
#endif
    }
}


CryptToken::~CryptToken() {
    SecureZeroMemory(token_safe_, token_size_);
    LocalFree(token_safe_);
}


char const* CryptToken::GetToken() const {
    return token_safe_;
}
} // namespace megamol::power
