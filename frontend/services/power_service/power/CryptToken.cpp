#include "CryptToken.h"

#include <filesystem>
#include <fstream>

#if WIN32
#include <Windows.h>
#include <wincred.h>
#include <wincrypt.h>
#pragma comment(lib, "Credui.lib")
#pragma comment(lib, "Crypt32.lib")
#endif

namespace megamol::power {
CryptToken::CryptToken(std::string const& filename) : token_(CREDUI_MAX_PASSWORD_LENGTH + 1), token_size_(0) {
//std::filesystem::path filepath(filename);
#if WIN32
    if (std::filesystem::exists(filename) && std::filesystem::is_regular_file(filename)) {
        token_size_ = std::filesystem::file_size(filename);
        std::string file_data;
        file_data.resize(token_size_);
        auto in_file = std::ifstream(filename, std::ios::binary);
        in_file.read(file_data.data(), token_size_);
        in_file.close();
        DATA_BLOB blob_in;
        blob_in.cbData = token_size_;
        blob_in.pbData = (BYTE*) (file_data.data());
        DATA_BLOB blob_out;
        if (!CryptUnprotectData(&blob_in, nullptr, nullptr, nullptr, nullptr, 0, &blob_out)) {
            throw std::runtime_error("[CryptToken] Cannot decrypt data");
        }
        token_size_ = blob_out.cbData;
        std::copy(blob_out.pbData, blob_out.pbData + blob_out.cbData, token_.GetPtr());
        SecureZeroMemory(blob_out.pbData, blob_out.cbData);
        LocalFree(blob_out.pbData);
    } else {
        BOOL tkSave;
        auto res = CredUIPromptForCredentials(nullptr, "DarUSDataverse", nullptr, 0, (PSTR) "", 0, token_.GetPtr(),
            CREDUI_MAX_PASSWORD_LENGTH + 1, &tkSave,
            CREDUI_FLAGS_DO_NOT_PERSIST | CREDUI_FLAGS_EXCLUDE_CERTIFICATES | CREDUI_FLAGS_GENERIC_CREDENTIALS |
                CREDUI_FLAGS_PASSWORD_ONLY_OK | CREDUI_FLAGS_KEEP_USERNAME);

        DATA_BLOB blob_in;
        blob_in.cbData = strlen(token_.GetPtr());
        blob_in.pbData = (BYTE*) token_.GetPtr();
        DATA_BLOB blob_out;
        if (!CryptProtectData(&blob_in, nullptr, nullptr, nullptr, nullptr, 0, &blob_out)) {
            throw std::runtime_error("[CryptToken] Cannot encrypt data");
        }

        std::ofstream out_file(filename, std::ios::binary);
        out_file.write((char const*) blob_out.pbData, blob_out.cbData);
        out_file.close();

        SecureZeroMemory(blob_out.pbData, blob_out.cbData);
        LocalFree(blob_out.pbData);
    }
#endif
}


char const* CryptToken::GetToken() const {
    return token_.GetPtr();
}
} // namespace megamol::power
