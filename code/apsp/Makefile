HIPCC = hipcc
CXXFLAGS = -O2

TARGET = apsp

SRCS = main.cpp kernel.hip
HEADERS = main.h

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o
