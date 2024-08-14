#pragma once

#include <string>

#include "ProtectedMemory.h"

namespace megamol::power {

/**
 * @brief Class for storing the Dataverse API token securely.
 */
class CryptToken {
public:
    /**
     * @brief Ctor. Stores token in @c ProtectedMemory.
     * If file at @c filename exists, token will be read from there.
     * If the file does not exists, a prompt will ask for the token
     * and writes the file with the token encrypted with user credentials.
     * @param filename The filepath to the token file.
     */
    CryptToken(std::string const& filename);

    /**
     * @brief Dtor.
     */
    ~CryptToken() = default;

    /**
     * @brief Access the token.
     * @return The token.
     */
    char const* GetToken() const;

private:
    power::ProtectedMemory token_;

    std::size_t token_size_;
};
} // namespace megamol::power
