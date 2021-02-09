# https://developer.nvidia.com/blog/building-cuda-applications-cmake/

foreach(obj ${OBJECTS})
  get_filename_component(obj_ext ${obj} EXT)
  get_filename_component(obj_name ${obj} NAME_WLE)
  get_filename_component(obj_dir ${obj} DIRECTORY)

  if (obj_ext MATCHES ".ptx")
    set(args --const --padd 0 --type char --name embedded_${obj_name} ${obj})
    execute_process(
      COMMAND "${BIN2C}" ${args}
      WORKING_DIRECTORY ${obj_dir}
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error_var
    )
    file(WRITE "${COPY_DIR}/${obj_name}_embedded.c" "${output}")
  endif()
endforeach()
