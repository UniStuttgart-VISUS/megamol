if(UNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -DUNIX -pedantic -std=c99 -ldl")
    set(COMMON_CXX_FLAGS "-std=c++11 -fPIC -fno-strict-aliasing -no-ansi-alias -DNOMINMAX -DUNIX -D_GNU_SOURCE -D_LIN${BITS}")
    if (DISABLE_WARINGS)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -w")
    endif()


    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${COMMON_CXX_FLAGS} -DDEBUG -D_DEBUG -g -ggdb -ldl" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COMMON_CXX_FLAGS} -DNDEBUG -D_NDEBUG -O3 -g0 -ldl" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${COMMON_CXX_FLAGS} -DNDEBUG -D_NDEBUG -O3 -g -ldl" CACHE STRING "" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Ox /Gm /MP /W3 /std=c99 /Qopenmp" CACHE STRING "" FORCE)
    set(COMMON_CXX_FLAGS "/Gm /DNOMINMAX /W3 /Qopenmp /MP /GR")
    if (DISABLE_WARINGS)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -w")
    endif()

    set(INTEL_OPTIMIZATION_FLAGS "/Ob2 /O3 /Oi /Qparallel /GA /QaxSSE3,CORE-AVX2" CACHE STRING "icc-specific optimization options - check your architecture!")

    set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_CXX_FLAGS} /MDd /Od /DDEBUG /D_DEBUG /Zi /Zo /debug /debug:inline-debug-info" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_CXX_FLAGS} /MD /DNDEBUG /D_NDEBUG ${INTEL_OPTIMIZATION_FLAGS}" CACHE STRING "" FORCE)
    # /Qguide:1 generates no result file for vislibversioninfo?
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_CXX_FLAGS} /MD /DNDEBUG /D_NDEBUG ${INTEL_OPTIMIZATION_FLAGS} /Zi /Zo /debug /debug:inline-debug-info /Qopt-report:2" CACHE STRING "" FORCE)
endif()
