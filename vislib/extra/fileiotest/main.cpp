#include <iostream>
#include <cstdint>
#include <chrono>
#include <typeinfo>

#include "vislib/sys/File.h"
#include "vislib/sys/BufferedFile.h"
#include "vislib/sys/MemmappedFile.h"
#include "vislib/sys/sysfunctions.h"

template<class TF, class TC>
void write_formatted_text_file_test(const TC& path, uint64_t line_count) {
  std::cout << "Testing " << typeid(TF).name() << std::endl;
  std::cout << "Writing " << line_count << " formatted lines" << std::endl;

  std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
  
  TF file;
  if (!file.Open(path, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::CREATE_OVERWRITE)) {
    std::cerr << "Failed to create file " << path << std::endl;
    return;
  }
  
  for (uint64_t i = 0; i < line_count; ++i) {
    vislib::sys::WriteFormattedLineToFile(file, "Test %d\n", static_cast<double>(i) / static_cast<double>(line_count));
  }
  
  file.Close();

  std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
  
  std::cout << "finished in " << std::chrono::duration<double>(end_time - start_time).count() << " seconds" << std::endl;
}

template<class TF, class TC>
void write_text_file_test(const TC& path, uint64_t line_count) {
  std::cout << "Testing " << typeid(TF).name() << std::endl;
  std::cout << "Writing " << line_count << " lines" << std::endl;

  std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
  
  TF file;
  if (!file.Open(path, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::CREATE_OVERWRITE)) {
    std::cerr << "Failed to create file " << path << std::endl;
    return;
  }
  vislib::StringA line("Test-line\n");
  
  for (uint64_t i = 0; i < line_count; ++i) {
    file.Write(line.PeekBuffer(), 10);    
  }
  
  file.Close();

  std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
  
  std::cout << "finished in " << std::chrono::duration<double>(end_time - start_time).count() << " seconds" << std::endl;
}

template<class TF, class TC>
void read_file_test(const TC& path) {
  std::cout << "Testing " << typeid(TF).name() << std::endl;

  std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
  
  TF file;
  if (!file.Open(path, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
    std::cerr << "Failed to open file " << path << std::endl;
    return;
  }
  unsigned int buf_size = 1024; // * 1024
  unsigned char *buf = new unsigned char[buf_size];
  
  while (!file.IsEOF()) {
    file.Read(buf, buf_size);
  }
  
  delete[] buf;
  
  file.Close();

  std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
  
  std::cout << "finished in " << std::chrono::duration<double>(end_time - start_time).count() << " seconds" << std::endl;
}

int main(int argc, char **argv) {
#ifdef _WIN32
#define TEST_PATH "D:\\tmp\\"
#else
#define TEST_PATH "/tmp/sgrottel/"
#endif
    std::cout << "Starting tests on file writing speed:" << std::endl;
    std::cout << std::endl;
    
    const uint64_t size1 = 100000ull;
    const uint64_t size2 = 10000000ull;
       
    write_text_file_test<vislib::sys::File>(TEST_PATH "vislib_iotest_file2.txt", size1);
    write_formatted_text_file_test<vislib::sys::File>(TEST_PATH "vislib_iotest_file.txt", size1);
    std::cout << std::endl;
    
    write_text_file_test<vislib::sys::BufferedFile>(TEST_PATH "vislib_iotest_buffedfile2.txt", size2);
    write_formatted_text_file_test<vislib::sys::BufferedFile>(TEST_PATH "vislib_iotest_buffedfile.txt", size2);
    std::cout << std::endl;

    write_text_file_test<vislib::sys::MemmappedFile>(TEST_PATH "vislib_iotest_memmapfile2.txt", size2);
    write_formatted_text_file_test<vislib::sys::MemmappedFile>(TEST_PATH "vislib_iotest_memmapfile.txt", size2);
    std::cout << std::endl;
    
    std::cout << "Starting tests on file reading speed:" << std::endl;
    std::cout << std::endl;
       
    read_file_test<vislib::sys::File>(TEST_PATH "vislib_iotest_buffedfile2.txt");
    std::cout << std::endl;
    
    read_file_test<vislib::sys::BufferedFile>(TEST_PATH "vislib_iotest_buffedfile2.txt");
    std::cout << std::endl;

    read_file_test<vislib::sys::MemmappedFile>(TEST_PATH "vislib_iotest_buffedfile2.txt");
    std::cout << std::endl;
    
    std::cout << "Finished" << std::endl;

    return 0;
}
