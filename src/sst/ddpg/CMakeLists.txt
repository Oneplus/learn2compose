include_directories (. ${PROJECT_SOURCE_DIR}/src/ ${PROJECT_SOURCE_DIR}/src/sst/)
add_executable (ddpg_sst
	main_ddpg_sst.cc
	ddpg_sst_model.cc
	ddpg_sst_model.h
	)

if (MSVC)
	target_link_libraries (ddpg_sst dynet common sst_common ${Boost_LIBRARIES} ${LIBS})
else()
	target_link_libraries (ddpg_sst dynet common sst_common	 ${Boost_LIBRARIES} ${LIBS})
endif()
