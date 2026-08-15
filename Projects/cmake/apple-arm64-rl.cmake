set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architecture" FORCE)
set(RL_NATIVE_ARM64_TRAINING ON CACHE BOOL "Build the stripped Apple Silicon RL training executable" FORCE)
set(ENABLE_STEAMWORKS OFF CACHE BOOL "Build with Steamworks Support" FORCE)
set(BREAKPAD OFF CACHE BOOL "Build with breakpad" FORCE)
