include_directories ()
add_library (common
	corpus.h
	corpus.cc
	alphabet.cc
	alphabet.h
	logging.cc
	logging.h
	layer.cc
	layer.h
	model.cc
	model.h
	system.cc
	system.h
	treelstm.cc
	treelstm.h
	trainer_utils.cc
	trainer_utils.h
	)

add_subdirectory(l2c.snli)
add_subdirectory(l2c.sst)
add_subdirectory(l2c.yelp)