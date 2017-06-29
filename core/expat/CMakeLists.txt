project(expat)

# configuration options
set(XML_CONTEXT_BYTES 1024)
set(XML_DTD ON)
set(XML_NS ON)

if(XML_DTD)
  set(XML_DTD 1)
else(XML_DTD)
  set(XML_DTD 0)
endif(XML_DTD)
if(XML_NS)
  set(XML_NS 1)
else(XML_NS)
  set(XML_NS 0)
endif(XML_NS)

set(EXPAT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" CACHE INTERNAL "")

include_directories(${EXPAT_INCLUDE_DIRS})


set(expat_SRCS
  src/xmlparse.c
  src/xmlrole.c
  src/xmltok.c
  src/xmltok_impl.c
  src/xmltok_ns.c
  )


add_library(expat STATIC ${expat_SRCS})
if(MSVC)
  set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/ignore:4996")
  target_compile_definitions(${PROJECT_NAME} PRIVATE XML_UNICODE_WCHAR_T XML_LARGE_SIZE)
endif(MSVC)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER external)

