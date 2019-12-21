cmake_minimum_required(VERSION 3.0)

project(bhtsne CXX)

set(BHTSNE_PUBLIC_HDRS
  tsne.h)
set(BHTSNE_PRIVATE_HDRS
  sptree.h
  vptree.h)
set(BHTSNE_SRCS
  tsne.cpp
  sptree.cpp)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(bhtsne STATIC ${BHTSNE_SRCS} ${BHTSNE_PUBLIC_HDRS} ${BHTSNE_PRIVATE_HDRS})

install(TARGETS bhtsne DESTINATION "lib")
install(FILES ${BHTSNE_PUBLIC_HDRS} DESTINATION "include")
