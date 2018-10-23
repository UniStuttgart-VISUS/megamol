project(AntTweakBar)

set(AntTweakBar_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")

file(GLOB_RECURSE SOURCES src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_compile_definitions(${PROJECT_NAME} PRIVATE TW_EXPORTS)
target_include_directories(${PROJECT_NAME} PUBLIC ${AntTweakBar_INCLUDE_DIR})

if(UNIX)
  target_compile_definitions(${PROJECT_NAME} PRIVATE _UNIX)
endif()

#target_link_libraries(${PROJECT_NAME} GL)

if(WIN32)
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
  install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
else()
  install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
endif()

set(AntTweakBar_LIBRARIES ${PROJECT_NAME} CACHE INTERNAL "")
