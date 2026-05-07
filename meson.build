project('Conemap', ['cpp', 'cuda'])

cuda_args = [
  '-diag-suppress=780',
  '-Wno-deprecated-gpu-targets',

  # required fix
  '-Xcompiler=/Zc:preprocessor',

  # split flags properly
  '-Xcompiler=/std:c++20',
  '-Xcompiler=/permissive-',

  # IMPORTANT: disable device debug
  # (remove -G entirely)
]

add_project_arguments(cuda_args, language: 'cuda')

cxxopts_dep = dependency('cxxopts', fallback: ['cxxopts', 'cxxopts_dep'])

conemap_lib = library(
  'conemap',
  'src/conemap.cu',
  include_directories: include_directories('src')
)

conemap_dep = declare_dependency(
  link_with : conemap_lib,
  include_directories: include_directories('src'),
)

# Build the standalone executable only when not a subproject
if not meson.is_subproject()
  executable(
    'Conemap',
    'src/main.cpp',
    link_with : conemap_lib,
    dependencies : cxxopts_dep,
    install : true,
  )
endif
