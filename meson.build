project('Conemap', ['cpp', 'cuda'])

add_project_arguments(
  [
  '-diag-suppress=780',
  '-Wno-deprecated-gpu-targets'
  ],
  language: 'cuda'
)

conemap_lib = library(
  'conemap',
  'src/conemap.cu',
  'src/kernels.cu',
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
    install : true,
  )
endif
