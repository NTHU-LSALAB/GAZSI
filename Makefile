# Makefile for Performance Test Client
# Compiles the C++ test client for GPUNetIO vs Python+CUDA comparison

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pthread
INCLUDES = -I.
LIBS = -lcurl -lpthread -lm

# Source files
SOURCES = performance_client.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = performance_client

# Default target
all: $(TARGET)

# Build the performance client
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LIBS)
	@echo "Build complete: $(TARGET)"

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -f results_*.json
	@echo "Clean complete"

# Run basic test
test: $(TARGET)
	@echo "Running basic connectivity test..."
	./$(TARGET) --test

# Run full performance comparison
benchmark: $(TARGET)
	@echo "Running full performance benchmark..."
	./$(TARGET)

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y libcurl4-openssl-dev build-essential

.PHONY: all clean test benchmark install-deps