vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO LTLA/CppKmeans
    REF 24a2a517f6a87b286c2395bb789211f0bedfe021
    SHA512 c4b816fb67a84d701c837aaacc5433a58395055f1521dd631538675c565d35b18dbadc351b3d9949c209909737725ff137c9c58c94d4caed441bd7cfcd960361
    HEAD_REF master
)

file(COPY "${SOURCE_PATH}/include/kmeans" DESTINATION "${CURRENT_PACKAGES_DIR}/include/ltla")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/ltla_cppkmeansConfig.cmake" DESTINATION "${CURRENT_PACKAGES_DIR}/share/ltla_cppkmeans")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
