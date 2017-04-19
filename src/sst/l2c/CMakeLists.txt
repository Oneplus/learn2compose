include_directories (. ${PROJECT_SOURCE_DIR}/src/ ${PROJECT_SOURCE_DIR}/src/sst/)
add_executable (learn2compose_sst
  main_l2c_sst.cc
  l2c_sst_model.cc
  l2c_sst_model.h
)

if (MSVC)
	target_link_libraries (learn2compose_sst dynet common sst_common ${Boost_LIBRARIES} ${LIBS})
else()
	target_link_libraries (learn2compose_sst dynet common sst_common ${Boost_LIBRARIES} ${LIBS})
endif()
