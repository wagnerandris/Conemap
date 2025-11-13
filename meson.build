project('Conemap', ['cpp', 'cuda'])

boost_dep = dependency('boost', modules : ['program_options'], required : true)

add_project_arguments(
  [
  '-diag-suppress=780',
  '-Wno-deprecated-gpu-targets'
  ],
  language: 'cuda'
)

executable('Conemap',
           'src/main.cpp',
           'src/conemap.cu',
           'src/kernels.cu',
           dependencies : [boost_dep],
           install : true)
