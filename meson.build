project('Conemap', ['cpp', 'cuda'])

boost_dep = dependency('boost', modules : ['program_options'], required : true)

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
  dependencies : [boost_dep],
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
    dependencies : [boost_dep],
    install : true,
  )
endif
