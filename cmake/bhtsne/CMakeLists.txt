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

add_library(bhtsne SHARED ${BHTSNE_SRCS} ${BHTSNE_PUBLIC_HDRS} ${BHTSNE_PRIVATE_HDRS})

install(FILES ${BHTSNE_PUBLIC_HDRS} DESTINATION "include")
if(WIN32)
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
  install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
else()
  install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
endif()
