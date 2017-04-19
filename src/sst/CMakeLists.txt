include_directories (. ${PROJECT_SOURCE_DIR}/src/)

add_library (sst_common
  sst_corpus.h
  sst_corpus.cc
  sst_corpus_with_tree.h
  sst_corpus_with_tree.cc
  sst_model_i.h
  sst_model_i.cc
  evaluate.h
  evaluate.cc)

target_link_libraries (sst_common common)

add_subdirectory (treelstm)
add_subdirectory (l2c)
add_subdirectory (pg)
add_subdirectory (ddpg)
