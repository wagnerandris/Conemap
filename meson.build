project('Conemap', ['cpp', 'cuda'])

add_project_arguments(
  ['-diag-suppress=780'],
  language: 'cuda'
)

executable('Conemap',
           'src/main.cu',
           install : true)
