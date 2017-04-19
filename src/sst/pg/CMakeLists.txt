include_directories (. ${PROJECT_SOURCE_DIR}/src/ ${PROJECT_SOURCE_DIR}/src/sst/)
add_executable (policygradient_sst
	main_pg_sst.cc
	pg_sst_model.cc
	pg_sst_model.h
	)

if (MSVC)
	target_link_libraries (policygradient_sst dynet common sst_common ${Boost_LIBRARIES} ${LIBS})
else()
	target_link_libraries (policygradient_sst dynet common sst_common	 ${Boost_LIBRARIES} ${LIBS})
endif()
