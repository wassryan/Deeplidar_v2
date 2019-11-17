XX=g++
LDFLAGS=-std=c++11 -O1 -g -w -fopenmp
TARGET=count_main.app

CFLAGS = -lstdc++

LIBS=`pkg-config --cflags --libs opencv`


# select use which tracker
WHICH_TRACKER=tracking

INCLUDE_BASE = ./

INCLUDE_DIR = -I"$(INCLUDE_BASE)"
	  
SRC_BASE = ./

SRCS := $(wildcard $(SRC_BASE)/*.cpp)

OBJS := $(SRCS:%.cpp=%.o)

all:$(OBJS)
	@echo "Compilling......"
	$(CXX) $(LDFLAGS) -o $(TARGET) $(OBJS) $(CFLAGS) $(LIBS)

%.o:%.cpp
	$(CXX) $(LDFLAGS) -c $^ -o $@ $(LIBS) $(CFLAGS) $(INCLUDE_DIR)
	@echo " "
	
.PHONY:clean
clean:
	rm -rf $(PROGRAM) $(OBJS) $(TARGET) sc.txt license.txt

