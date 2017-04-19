include_directories (. ${PROJECT_SOURCE_DIR}/src/ ${PROJECT_SOURCE_DIR}/src/sst/)
add_executable (treelstm_sst
  main_treelstm_sst.cc
  treelstm_sst_model.cc
  treelstm_sst_model.h
)

if (MSVC)
  target_link_libraries (treelstm_sst dynet common sst_common ${Boost_LIBRARIES} ${LIBS})
else()
  target_link_libraries (treelstm_sst dynet common sst_common ${Boost_LIBRARIES} ${LIBS})
endif()
