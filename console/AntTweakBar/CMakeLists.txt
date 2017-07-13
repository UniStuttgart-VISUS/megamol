project(AntTweakBar)

set(AntTweakBar_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")

include_directories(${AntTweakBar_INCLUDE_DIR})

file(GLOB_RECURSE SOURCES src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_compile_definitions(${PROJECT_NAME} PRIVATE TW_EXPORTS)

if(UNIX)
  target_compile_definitions(${PROJECT_NAME} PRIVATE UNIX)
endif()

#target_link_libraries(${PROJECT_NAME} GL)

if(WIN32)
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
  install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX "64.lib")
  endif()
else()
  install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
endif()

set(AntTweakBar_LIBRARIES ${PROJECT_NAME} CACHE INTERNAL "")

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER external)